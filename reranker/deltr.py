from collections import Counter

import fairsearchdeltr
import numpy as np
import pandas as pd
from fairsearchdeltr.deltr import prepare_data
from fairsearchdeltr.trainer import Trainer, topp_prot, find_items_per_group_per_query, normalized_exposure
from tqdm import tqdm

import src.reranker.model as model
from features.features import AnnotationFeatureEngineer


# class NanToZeroTrainer(Trainer):
#     def _exposure_diff(self, data, query_ids, which_query, prot_idx):
#         """
#         computes the exposure difference between protected and non-protected groups
#         implementation of equation 5 in DELTR paper but without the square
#
#         :param data: all predictions
#         :param query_ids: list of query IDs
#         :param which_query: given query ID
#         :param prot_idx: list states which item is protected or non-protected
#
#         :return: float value
#         """
#         judgments_per_query, protected_items_per_query, nonprotected_items_per_query = \
#             find_items_per_group_per_query(data, query_ids, which_query, prot_idx)
#
#         exposure_prot = normalized_exposure(protected_items_per_query,
#                                             judgments_per_query)
#         exposure_nprot = normalized_exposure(nonprotected_items_per_query,
#                                              judgments_per_query)
#
#         expdiff = (exposure_nprot - exposure_prot)
#
#         if np.isnan(expdiff):
#             expdiff = 0
#
#         exposure_diff = np.maximum(0, expdiff)
#
#         return exposure_diff
#
# class NanToZeroDeltr(fairsearchdeltr.Deltr):
#     def train(self, training_set: pd.DataFrame):
#         """
#         Trains a DELTR model on a given training set
#         :param training_set:        requires first column to contain the query ids, second column the document ids
#                                     and last column to contain the training judgements in descending order
#                                     i.e. higher scores are better
#         :return:                    returns the model
#         """
#
#         # find the protected feature index
#         names = training_set.columns.tolist()
#         if self._protected_feature_name in names:
#             # the first 2 columns should ALWAYS be query id and document id, that's why we subtract 2
#             self._protected_feature = names.index(self._protected_feature_name) - 2
#         else:
#             raise ValueError("The name of the protected feature does not appear in the `DataFrame`")
#
#         # create the trainer
#         tr = NanToZeroTrainer(self._protected_feature, self._gamma, self._number_of_iterations, self._learning_rate,
#                              self._lambda, self._init_var)
#
#         # prepare data
#         query_ids, doc_ids, protected_attributes, feature_matrix, training_scores = prepare_data(training_set,
#                                                                                                  self._protected_feature)
#
#         # standardize data if allowed
#         if self._standardize:
#             self._mus = feature_matrix.mean()
#             self._sigmas = feature_matrix.std()
#             protected_feature = feature_matrix[:,self._protected_feature]
#             feature_matrix = (feature_matrix - self._mus) / self._sigmas
#             feature_matrix[:, self._protected_feature] = protected_feature
#
#         # launch training routine
#         self._omega, self._log = self._train_nn(tr, query_ids, feature_matrix, training_scores)
#
#         # return model
#         return self._omega
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

    def __filter_queries_with_all_items_in_one_group(self, df):

        """
        When training, discard queries for which all items belong to a single group or for which one or more groups
        only have one item. These cause division by zero exceptions during the training process.
        :param g:
        :return:
        """
        c = Counter(df.protected.to_list())
        if len(c) == 1:
            return False
        return True

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
            # todo (BIG): yeet down elegant solution l8r
            x = x.groupby('q_num').filter(self.__filter_queries_with_all_items_in_one_group)

        x = x.reindex(columns=column_order)  # protected variables has to be at third position, somehow...
        return (x.drop_duplicates())

    def train(self, inputhandler):
        np.random.seed(self.random_state)
        data = self._prepare_data(inputhandler)
        weights = self.dtr1.train(data)
        if self.dtr2:
            np.random.seed(self.random_state)
            weights2 = self.dtr2.train(data)
            return weights, weights2
        return weights

    def __apply_rank(self, df, ranker, has_judgment=False):
        ranking_columns = df.columns[1:]
        ranking = ranker.rank(df[ranking_columns], has_judgment)
        # assert len(df) == len(ranking)
        # assert (df[['doc_id', 'protected']].sort_values(by=['doc_id', 'protected']) == ranking[
        #     ['doc_id', 'protected']].sort_values(by=['doc_id', 'protected'])).all().all()
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

        tqdm.pandas()
        print("Ranker 1 ranking...")
        data1 = data1.groupby(['sid', 'q_num']).progress_apply(self.__apply_rank, ranker=self.dtr1, has_judgment=False)
        data1 = data1.reset_index(drop=True)[['sid', 'q_num', 'doc_id', 'judgement']]

        if self.dtr2:
            print("Ranker 2 ranking...")
            data2 = data.copy()
            data2 = data2.groupby(['sid', 'q_num']).progress_apply(self.__apply_rank, ranker=self.dtr2, has_judgment=False)
            data2 = data2.reset_index(drop=True)[['sid', 'q_num', 'doc_id', 'judgement']]

            m = pd.merge(data1, data2, on=['sid', 'q_num', 'doc_id'])
            print("Merging judgements...")
            m['judgement'] = m.progress_apply(lambda row: self.alpha * row.judgement_x + (1 - self.alpha) * row.judgement_y,
                                     axis=1)

            data = m


        else:
            data = data1

        print("Converting judgements to rankings...")
        data['rank'] = data.groupby(['sid', 'q_num'])['judgement'].progress_apply(pd.Series.rank, ascending=False,
                                                                         method='first')

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data,
                        how='left', on=['sid', 'q_num',
                                        'doc_id'])  # query seq is in first here b/c we want to rank each item for each query

        return pred
