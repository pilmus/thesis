from fairsearchdeltr import Deltr
import pandas as pd
import src.reranker.model as model
from features.features import DeltrFeatureEngineer


class DeltrWrapper(model.RankerInterface):
    """
    Wrapper arround the Deltr algorithm without any fairness enabled.
    """

    def __init__(self, featureengineer: DeltrFeatureEngineer, protected_feature, group_mapping):
        super().__init__(featureengineer)
        # setup the DELTR object
        # protected_feature = 'in_first'  # column name of the protected attribute (index after query and document id)
        gamma = 0  # value of the gamma parameter
        number_of_iterations = 5  # number of iterations the training should run
        standardize = True  # let's apply standardization to the features

        # create the Deltr object
        self.protected_feature = protected_feature
        self.group_mapping = group_mapping
        self.dtr = Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)
        self.weights = None

    def __apply_grouping(self, value):
        return self.group_mapping.get(value)

    def _prepare_data(self, inputhandler, has_judgment=True):
        """
        requires first column to contain the query ids, second column the document ids and last column to contain the training judgements in descending order i.e. higher scores are better
        """
        column_order = ["q_num", "doc_id", "protected",
                        "title_score", "abstract_score", "entities_score",
                        "venue_score", "journal_score", "authors_score", "inCitations", "outCitations"
                                                                                        "qlength"]

        features = self.fe.get_feature_mat(inputhandler)
        data = inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id', 'relevance']]
        data = pd.merge(data, features, how='left', on=['qid', 'doc_id'])
        data['protected'] = data[self.protected_feature].apply(self.__apply_grouping)

        if not has_judgment:
            data.drop('relevance', axis=1, inplace=True)
        else:
            column_order.append("relevance")

        data.dropna(inplace=True)  # drop missing values as some doc_ids are not in the corpus

        data = data.reindex(columns=column_order)  # protected variables has to be at third position, somehow...
        return (data.drop_duplicates())

    def train(self, inputhandler):
        data = self._prepare_data(inputhandler)
        self.weights = self.dtr.train(data)
        return (self.weights)

    def _predict(self, inputhandler):
        """
        requires first column to contain the query ids, second column the document ids
                                    and (optionally) last column to contain initial judgements in descending order
                                    i.e. higher scores are better
        """

        data = self._prepare_data(inputhandler, has_judgment=False)
        data = data.groupby(['sid', 'q_num']).apply(self.dtr.rank, has_judgment=False)
        data.reset_index(inplace=True, level=0)
        data['rank'] = data.groupby(['sid', 'q_num'])['judgement'].apply(pd.Series.rank, ascending=False,
                                                                         method='first')

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data,
                        how='left', on=['sid', 'q_num', 'doc_id'])

        return pred
