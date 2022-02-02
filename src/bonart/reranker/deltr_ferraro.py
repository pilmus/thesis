import pickle

import pandas as pd
from fairsearchdeltr import Deltr

import src.bonart.reranker.model as model


class DeltrFerraro(model.RankerInterface):
    """
    Wrapper arround DELTR, contains two separate DELTR models trained on the same dataset whose scores are combined in the prediction phase.
    """

    COLUMN_ORDER = ["q_num", "doc_id", "group",
                    "abstract_score", "authors_score", "entities_score",
                    "inCitations", "journal_score", "outCitations", "title_score",
                    "venue_score", "qlength"]

    def __init__(self, featureengineer, group_file, standardize=False, relevance_weight=0.25, iter_nums = 5):
        super().__init__(featureengineer)
        # setup the DELTR object
        self.standardize = standardize
        self.iter_nums = iter_nums

        # create the Deltr object
        self.dtr_zero = Deltr("group", 0, number_of_iterations=iter_nums, standardize=standardize)
        self.dtr_one = Deltr("group", 1, number_of_iterations=iter_nums, standardize=standardize)

        self._grouping = pd.read_csv(group_file)
        self.relevance_weight = relevance_weight

    def __grouping_apply(self, df):
        df = pd.merge(df, self.grouping[['doc_id', 'group']], how='left', on='doc_id')

        return df

    def __is_unique(self, s):
        a = s.to_numpy()  # s.values (pandas<0.24)
        return (a[0] == a).all()

    def __prepare_data(self, inputhandler, has_judgment=True, mode='train'):
        """
        DELTR requires the data to be in a specific order: qid, docid, protected feature (group), ...
        """
        print(f"Preparing data...")
        print(f"Getting features...")
        features = self.fe.get_feature_mat(inputhandler)

        print("Rest of the prep...")
        data = inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id', 'relevance']]

        data = pd.merge(data, features, how='left', on=['qid', 'doc_id'])

        data = data.groupby('qid', as_index=False).apply(self.__grouping_apply)

        # remove queries for which all documents belong to one group -- this leads to nan values in training
        if mode == 'train':
            data = data.groupby('qid').filter(lambda g: not self.__is_unique(g.group))

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
        # annotations

        data = data.reindex(columns=col_order)  # protected variable has to be at third position for DELTR

        data = data.drop_duplicates()
        return data

    def train(self, inputhandler):
        data = self.__prepare_data(inputhandler)
        print(f"Training gamma == 1...")
        self.dtr_one.train(data)

        print(f"Training gamma == 0...")
        self.dtr_zero.train(data)

        return self.dtr_one, self.dtr_zero

    def _prepare_data(self, inputhandler, has_judgment, mode):
        return self.__prepare_data(inputhandler, has_judgment, mode)

    def _predict(self, inputhandler):
        """
        requires first column to contain the query ids, second column the document ids
                                    and (optionally) last column to contain initial judgements in descending order
                                    i.e. higher scores are better
        """

        data = self.__prepare_data(inputhandler, has_judgment=False, mode='eval')

        print(f"Ranking zero...")
        data_zero = data.groupby('q_num').apply(self.dtr_zero.rank, has_judgment=False)
        data_zero = data_zero.rename({'judgement': 'judgement_zero'}, axis=1)

        print(f"Ranking one...")
        data_one = data.groupby('q_num').apply(self.dtr_one.rank, has_judgment=False)
        data_one = data_one.rename({'judgement': 'judgement_one'}, axis=1)

        data_merged = pd.merge(data_one.reset_index(), data_zero.reset_index(), on=['q_num', 'doc_id'])
        data_merged['judgement'] = data_merged.apply(
            lambda row: self.relevance_weight * row.judgement_zero + (1 - self.relevance_weight) * row.judgement_one,
            axis=1)

        # data_mean_judgment = \
        # pd.concat((data_one.reset_index(), data_zero.reset_index())).groupby(['q_num', 'doc_id'], as_index=False,
        #                                                                      sort=False)[
        #     'judgement'].mean()

        data = pd.merge(data,
                        data_merged[['q_num', 'doc_id', 'judgement']],
                        on=['q_num', 'doc_id'], how='left')

        # data = data.reset_index(level=0) #necessary?

        data['rank'] = data.groupby('q_num')['judgement'].apply(pd.Series.rank, ascending=False,
                                                                method='first')

        data[['sid', 'q_num']] = data['q_num'].str.split('.', expand=True)

        data = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data, how='left',
                        on=['sid', 'q_num', 'doc_id'])

        return data

    def save(self):
        print(f"Saving models...")
        zero_path = f"resources/models/2020/deltr_gamma_0_std_{self.standardize}_iter_{self.iter_nums}.pickle"
        one_path = f"resources/models/2020/deltr_gamma_1_std_{self.standardize}_iter_{self.iter_nums}.pickle"

        with open(zero_path, 'wb') as fp:
            pickle.dump(self.dtr_zero, fp)
        with open(one_path, 'wb') as fp:
            pickle.dump(self.dtr_one, fp)
        return zero_path, one_path

    def load(self, dtr_zero_path, dtr_one_path):
        with open(dtr_zero_path, "rb") as fp:
            self.dtr_zero = pickle.load(fp)
        with open(dtr_one_path, "rb") as fp:
            self.dtr_one = pickle.load(fp)
        return True

    @property
    def grouping(self):
        return self._grouping

    @grouping.setter
    def grouping(self, value):
        self._grouping = pd.read_csv(value)
