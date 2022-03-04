import fairsearchdeltr
import numpy as np
import pandas as pd
from tqdm import tqdm

import src.reranker.model as model
from features.features import AnnotationFeatureEngineer


class Deltr(model.RankerInterface):
    """
    Wrapper arround the Deltr algorithm.
    """

    def __init__(self, featureengineer: AnnotationFeatureEngineer, protected_feature, group_mapping, gamma1, num_iter=5,
                 random_state=None, gamma2=None, alpha=None):
        super().__init__(featureengineer)
        # setup the DELTR object
        # protected_feature = 'in_first'  # column name of the protected attribute (index after query and document id)
        number_of_iterations = num_iter  # number of iterations the training should run
        standardize = True  # let's apply standardization to the features

        # create the Deltr object
        self.protected_feature = protected_feature
        self.group_mapping = group_mapping

        self.dtr1 = fairsearchdeltr.Deltr("protected", gamma1, number_of_iterations, standardize=standardize)
        self.dtr2 = None
        self.alpha = None
        if gamma2 is not None:
            if alpha is None:
                raise ValueError(
                    f"If you want to use two rankers you have to specify a linear interpolation parameter.")
            self.alpha = alpha
            self.dtr2 = fairsearchdeltr.Deltr("protected", gamma2, number_of_iterations, standardize=standardize)
        self.random_state = random_state

    def __apply_grouping(self, value):
        return self.group_mapping[value]

    def _prepare_data(self, inputhandler, has_judgment=True):
        """
        requires first column to contain the query ids, second column the document ids and last column to contain the training judgements in descending order i.e. higher scores are better
        """
        column_order = ["q_num", "doc_id", "protected",
                        "title_score", "abstract_score", "entities_score",
                        "venue_score", "journal_score", "authors_score", "inCitations", "outCitations",
                        "qlength"]

        x = self.fe.get_feature_mat(inputhandler, [self.protected_feature])
        y = inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id', 'relevance']]
        x = pd.merge(x, y, how='left', on=['qid', 'doc_id'])

        tqdm.pandas()
        x['protected'] = x[self.protected_feature].progress_apply(self.__apply_grouping)

        if not has_judgment:
            x.drop('relevance', axis=1, inplace=True)
            column_order = ['sid'] + column_order  # if we're in eval mode we're gonna need the sid
        else:
            column_order.append("relevance")

        x = x.reindex(columns=column_order)  # protected variables has to be at third position, somehow...
        return (x.drop_duplicates())

    def train(self, inputhandler):
        np.random.seed(self.random_state)
        data = self._prepare_data(inputhandler)
        weights = self.dtr1.train(data)
        if self.dtr2:
            weights2 = self.dtr2.train(data)
            return weights, weights2
        return weights

    def __apply_rank(self, df, ranker, has_judgment=False):
        ranking_columns = df.columns[1:]
        ranking = ranker.rank(df[ranking_columns], has_judgment)
        assert len(df) == len(ranking)
        assert (df.protected.to_list() == ranking.protected.to_list())
        df = pd.merge(df, ranking, on=['doc_id', 'protected'])
        return df

    def _predict(self, inputhandler):
        """
        requires first column to contain the query ids, second column the document ids
                                    and (optionally) last column to contain initial judgements in descending order
                                    i.e. higher scores are better
        """

        data = self._prepare_data(inputhandler, has_judgment=False)
        data1 = data.copy()

        data1 = data1.groupby(['sid', 'q_num']).apply(self.__apply_rank, self.dtr1, has_judgment=False)
        data1 = data1.reset_index(drop=True)[['sid', 'q_num', 'doc_id', 'judgement']]

        if self.dtr2:
            data2 = data.copy()
            data2 = data2.groupby(['sid', 'q_num']).apply(self.__apply_rank, self.dtr2, has_judgment=False)
            data2 = data2.reset_index(drop=True)[['sid', 'q_num', 'doc_id', 'judgement']]

            m = pd.merge(data1, data2, on=['sid', 'q_num', 'doc_id'])
            m['judgement'] = m.apply(lambda row: self.alpha * row.judgement_x + (1 - self.alpha) * row.judgement_y,
                                     axis=1)

            data = m


        else:
            data = data1

        data['rank'] = data.groupby(['sid', 'q_num'])['judgement'].apply(pd.Series.rank, ascending=False,
                                                                         method='first')

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data,
                        how='left', on=['sid', 'q_num',
                                        'doc_id'])  # query seq is in first here b/c we want to rank each item for each query

        return pred
