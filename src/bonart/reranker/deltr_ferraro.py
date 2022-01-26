import pickle

import pandas as pd
from fairsearchdeltr import Deltr

import src.bonart.reranker.model as model


class DeltrFerraro(model.RankerInterface):
    """
    Wrapper arround DELTR, contains two separate DELTR models trained on the same dataset whose scores are combined in the prediction phase.
    """

    COLUMN_ORDER = ["q_num", "doc_id", "protected",
                    "abstract_score", "authors_score", "entities_score",
                    "inCitations", "journal_score", "outCitations", "title_score",
                    "venue_score", "qlength"]

    def __init__(self, featureengineer, protected_feature, protected_feature_mapping, group_file, standardize=False):
        super().__init__(featureengineer)
        # setup the DELTR object
        self._protected_feature = protected_feature
        self._protected_feature_mapping = protected_feature_mapping

        # create the Deltr object
        self.dtr_zero = Deltr("protected", 0, number_of_iterations=5, standardize=standardize)
        self.dtr_one = Deltr("protected", 1, number_of_iterations=5, standardize=standardize)

        self._grouping = pd.read_csv(group_file)


    def __grouping_apply(self, df):

        # todo: warning if not two groups??
        self.grouping['protected'] = self.grouping[self.protected_feature].map(self.protected_feature_mapping)
        df = pd.merge(df,self.grouping[['doc_id','protected']], how='left',on='doc_id')

        # df['protected'] = self.grouping[self.protected_feature].map(self.protected_feature_mapping)
        return df

    def __prepare_data(self, inputhandler, has_judgment=True, mode='train'):
        """
        DELTR requires the data to be in a specific order: qid, docid, protected feature, ...
        """
        print(f"Preparing data...")
        print(f"Getting features...")
        features = self.fe.get_feature_mat(inputhandler)

        print("Rest of the prep...")
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
        print(f"Training gamma == 0...")
        self.dtr_zero.train(data)

        print(f"Training gamma == 1...")
        self.dtr_one.train(data)

        return self.dtr_one, self.dtr_zero

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

        data = data.groupby('q_num').apply(self.dtr_zero.rank, has_judgment=False)
        data = data.reset_index(level=0)

        data['rank'] = data.groupby('q_num')['judgement'].apply(pd.Series.rank, ascending=False,
                                                                method='first')

        data[['sid', 'q_num']] = data['q_num'].str.split('.', expand=True)

        data = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data, how='left',
                        on=['sid', 'q_num', 'doc_id'])

        return data

    def save(self):
        print(f"Saving models...")
        with open(f"resources/models/2020/deltr_gamma_0_prot_{self._protected_feature}.pickle",
                  'wb') as fp:
            pickle.dump(self.dtr_zero, fp)
        with open(f"resources/models/2020/deltr_gamma_1_prot_{self._protected_feature}.pickle",
                  'wb') as fp:
            pickle.dump(self.dtr_one, fp)
        return f"resources/models/2020/deltr_gamma_0_prot_{self._protected_feature}.pickle", f"resources/models/2020/deltr_gamma_1_prot_{self._protected_feature}.pickle"

    def load(self, dtr_zero_path, dtr_one_path):
        with open(dtr_zero_path, "rb") as fp:
            self.dtr_zero =   pickle.load(fp)
        with open(dtr_one_path, "rb") as fp:
            self.dtr_one =   pickle.load(fp)
        return True

    @property
    def protected_feature(self):
        return self._protected_feature

    @protected_feature.setter
    def protected_feature(self, value):
        self._protected_feature = value

    @property
    def protected_feature_mapping(self):
        return self._protected_feature_mapping

    @protected_feature_mapping.setter
    def protected_feature_mapping(self, value):
        self._protected_feature_mapping = value

    @property
    def grouping(self):
        return self._grouping

    @grouping.setter
    def grouping(self, value):
        self._grouping = pd.read_csv(value)