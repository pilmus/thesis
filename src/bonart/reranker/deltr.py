import pandas as pd
from fairsearchdeltr import Deltr

import src.bonart.reranker.model as model


class DeltrWrapper(model.RankerInterface):
    """
    Wrapper arround DELTR
    """

    COLUMN_ORDER = ["q_num", "doc_id", "protected",
                    "abstract_score", "authors_score", "entities_score",
                    "inCitations", "journal_score", "outCitations", "title_score",
                    "venue_score", "qlength"]

    def __init__(self, featureengineer, protected_feature_mapping, gamma, group_file, standardize=False):
        super().__init__(featureengineer)
        # setup the DELTR object
        self.protected_feature_name = protected_feature_mapping['feature_name']
        self.protected_feature_mapping = protected_feature_mapping

        # create the Deltr object
        self.dtr = Deltr("protected", gamma, number_of_iterations=5, standardize=standardize)

        self.doc_annotations = pd.read_csv(group_file)

    def __grouping_apply(self, df):  # todo: generic method?

        # todo: warning if not two groups
        self.doc_annotations['protected'] = self.doc_annotations.DocHLevel.map(self.protected_feature_mapping[
                                                                                   'value_mapping'])

        df['protected'] = self.doc_annotations['protected']
        return df

    def __prepare_data(self, inputhandler, has_judgment=True, mode='train'):
        """
        DELTR requires the data to be in a specific order: qid, docid, protected feature, ...
        """

        features = self.fe.get_feature_mat(inputhandler)

        data = inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id', 'relevance']]

        data = pd.merge(data, features, how='left', on=['qid', 'doc_id'])

        data = data.groupby('qid', as_index=False).apply(self.__grouping_apply)
        col_order = self.COLUMN_ORDER
        if has_judgment:
            col_order = self.COLUMN_ORDER + ['relevance']
        else:
            data = data.drop('relevance', axis=1)

        if mode == 'train':
            col_order[0] = 'qid'
        if mode == 'eval':
            data.q_num = data.sid.astype(str) + '.' + data.q_num.astype(str)

        data = data.dropna()  # drop missing values as some doc_ids are not in the corpus and not all docs have
        # Hlevel annotations

        data = data.reindex(columns=col_order)  # protected variable has to be at third position for DELTR

        data = data.drop_duplicates()
        return data

    def train(self, inputhandler):
        data = self.__prepare_data(inputhandler)
        self.weights = self.dtr.train(data)

        return self.weights

    def __predict_apply(self, df):

        df_copy = df.copy(deep=True)
        df_copy.q_num_combi = df.sid + "." + df.q_num
        df_copy = df_copy.drop(['sid', 'q_num'], axis=1)

        predictions = self.dtr.rank(df_copy, has_judgment=False)
        predictions[['sid', 'q_num']] = predictions['q_num'].str.split('.')
        return df

    def _predict(self, inputhandler):
        """
        requires first column to contain the query ids, second column the document ids
                                    and (optionally) last column to contain initial judgements in descending order
                                    i.e. higher scores are better
        """

        data = self.__prepare_data(inputhandler, has_judgment=False, mode='eval')

        data = data.groupby('q_num').apply(self.dtr.rank, has_judgment=False)
        data = data.reset_index(level=0)

        data['rank'] = data.groupby('q_num')['judgement'].apply(pd.Series.rank, ascending=False,
                                                                method='first')

        data[['sid', 'q_num']] = data['q_num'].str.split('.', expand=True)

        data = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data, how='left',
                        on=['sid', 'q_num', 'doc_id'])

        return data
