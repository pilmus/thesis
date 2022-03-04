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

    def __init__(self, featureengineer: AnnotationFeatureEngineer, protected_feature, group_mapping, gamma, num_iter=5,
                 random_state=None):
        super().__init__(featureengineer)
        # setup the DELTR object
        # protected_feature = 'in_first'  # column name of the protected attribute (index after query and document id)
        gamma = gamma  # value of the gamma parameter
        number_of_iterations = num_iter  # number of iterations the training should run
        standardize = True  # let's apply standardization to the features

        # create the Deltr object
        self.protected_feature = protected_feature
        self.group_mapping = group_mapping
        self.dtr = fairsearchdeltr.Deltr("protected", gamma, number_of_iterations, standardize=standardize)
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
        weights = self.dtr.train(data)
        return weights

    def __apply_rank(self, df, has_judgment=False):
        ranking_columns = df.columns[1:]
        ranking = self.dtr.rank(df[ranking_columns])
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
        data = data.groupby(['sid', 'q_num']).apply(self.__apply_rank, has_judgment=False)
        data.reset_index(inplace=True, drop=True)
        data['rank'] = data.groupby(['sid', 'q_num'])['judgement'].apply(pd.Series.rank, ascending=False,
                                                                         method='first')

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data,
                        how='left', on=['sid', 'q_num',
                                        'doc_id'])  # query seq is in first here b/c we want to rank each item for each query

        return pred
