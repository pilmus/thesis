import os
import pickle
from collections import Counter

import pandas as pd
from fairsearchdeltr import Deltr
from tqdm import tqdm

import src.bonart.reranker.model as model

# class Deltr(LibDeltr):
#     def _train_nn(self, tr, query_ids, feature_matrix, training_scores):
#         old_calc_cost = tr._calculate_cost
#         old_exposure_diff = tr._exposure_diff
#         new_query_ids = query_ids
#         wrapped_query_ids = tqdm(query_ids)
#
#         def new_calc_cost(training_judgments, predictions, _, prot_idx, data_per_query_predicted):
#             return old_calc_cost(training_judgments, predictions, wrapped_query_ids, prot_idx, data_per_query_predicted)
#
#         def new_exposure_diff(predictions, _, which_query, prot_idx):
#             return old_exposure_diff(predictions, new_query_ids, which_query, prot_idx)
#
#         tr._calculate_cost = new_calc_cost
#         tr._exposure_diff = new_exposure_diff
#
#         return tr.train_nn(query_ids, feature_matrix, training_scores)


class DeltrFerraro(model.RankerInterface):
    """
    Wrapper arround DELTR, contains two separate DELTR models trained on the same dataset whose scores are combined in the prediction phase.
    """

    COLUMN_ORDER = ["q_num", "doc_id", "group",
                    "abstract_score", "authors_score", "entities_score",
                    "in_citations", "journal_score", "out_citations", "title_score",
                    "venue_score", "qlength"]

    def __init__(self, featureengineer, group_file, group_name, standardize=False, alpha=0.25, iter_nums=5):
        """

        :param featureengineer:
        :param group_file:
        :param standardize:
        :param alpha: Determines the weight of relevance vs fairness. Lower alpha is more emphasis on relevance.
        :param iter_nums:
        """
        super().__init__(featureengineer)
        # setup the DELTR object
        self.standardize = standardize
        self.iter_nums = iter_nums
        self.group_name=group_name

        # create the Deltr object
        self.dtr_zero = Deltr("group", 0, number_of_iterations=iter_nums, standardize=standardize)
        self.dtr_one = Deltr("group", 1, number_of_iterations=iter_nums, standardize=standardize)
        self._grouping = pd.read_csv(group_file)
        self.alpha = alpha

    def __grouping_apply(self, df):
        df = pd.merge(df, self.grouping[['doc_id', 'group']], how='left', on='doc_id')
        return df

    def __groups_mult_members(self, g):
        """
        When training, discard queries for which all items belong to a single group or for which one or more groups
        only have one item. These cause division by zero exceptions during the training process.
        :param g:
        :return:
        """
        c = Counter(g.group.to_list())
        if len(c) == 1:
            return False
        # for v in c.values():
        #     if v == 1:
        #         continue
        #         return False
        return True

    def __prepare_data(self, inputhandler, has_judgment=True, mode='train'):
        """
        DELTR requires the data to be in a specific order: qid, docid, protected feature (group), ...
        """
        print(f"Preparing data...")
        features = self.fe.get_feature_mat_from_iohandler(inputhandler)

        print("Rest of the prep...")
        data = inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id', 'relevance']]

        data = pd.merge(data, features, how='left', on=['qid', 'doc_id'])

        data = data.groupby('qid', as_index=False).apply(self.__grouping_apply)

        # remove queries for which all documents belong to one group -- this leads to nan values in training
        if mode == 'train':
            data = data.groupby('qid').filter(self.__groups_mult_members)

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

    def __weight_judgements(self, row):
        # print(f"{row.judgement_zero} - {row.judgement_one}")
        return self.alpha * row.judgement_zero + (1 - self.alpha) * row.judgement_one

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
        # data_merged['judgement'] = data_merged.apply(
        #     lambda row: self.alpha * row.judgement_zero + (1 - self.alpha) * row.judgement_one,
        #     axis=1)

        data_merged['judgement'] = data_merged.apply(self.__weight_judgements,
                                                     axis=1)

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
        zero_path = f"resources/models/2020/deltr_gamma_0_alpha_{self.alpha}_corp_{self.fe.corpus.index}-group-{self.group_name}.pickle"
        one_path = f"resources/models/2020/deltr_gamma_1_alpha_{self.alpha}_corp_{self.fe.corpus.index}-group-{self.group_name}.pickle"

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
