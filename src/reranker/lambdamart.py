import random

import pyltr
import pandas as pd
from tqdm import tqdm

import reranker.model as model


class LambdaMart(model.RankerInterface):
    """
    Wrapper around the LambdaMart algorithm
    """

    def __init__(self, featureengineer, random_state=None):
        super().__init__(featureengineer)

        self.metric = pyltr.metrics.NDCG(k=7)

        self.lambdamart = pyltr.models.LambdaMART(
            metric=self.metric,
            n_estimators=1000,
            learning_rate=0.02,
            max_features=0.5,
            query_subsample=0.5,
            max_leaf_nodes=10,
            min_samples_leaf=64,
            verbose=1,
            random_state=random_state)

    def __data_helper(self, x):
        x = x.sort_values('q_num')
        qids = x[['sid', 'q_num', 'doc_id']]
        y = x['relevance']
        x.drop(['q_num', 'doc_id', 'relevance', 'qid'], inplace=True, axis=1)
        return (x, y, qids)

    def _prepare_data(self, inputhandler, frac=0.66, random_state=None, missing_value_strategy=None):
        x = self.fe.get_feature_mat(inputhandler, missing_value_strategy=missing_value_strategy)
        y = inputhandler.get_query_seq()[['sid', 'qid', "q_num", "doc_id", "relevance"]]
        x = pd.merge(x, y, how="left", on=['qid', 'doc_id'])
        training = x.q_num.drop_duplicates().sample(frac=frac, random_state=random_state)  # todo: unfix random state?
        x_train, y_train, qids_train = self.__data_helper(x.loc[x.q_num.isin(training)])

        x_val, y_val, qids_val = [None] * 3

        if frac < 1:
            x_val, y_val, qids_val = self.__data_helper(x.loc[~x.q_num.isin(training)])

        return (x_train, y_train, qids_train, x_val, y_val, qids_val)

    def train(self, inputhandler, random_state=None, missing_value_strategy=None):
        """
        X : array_like, shape = [n_samples, n_features] Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
        y : array_like, shape = [n_samples] Target values (integers in classification, real numbers in regression)
        For classification, labels must correspond to classes.
        qids : array_like, shape = [n_samples] Query ids for each sample. Samples must be grouped by query such that
        all queries with the same qid appear in one contiguous block.
        """

        x_train, y_train, qids_train, x_val, y_val, qids_val = self._prepare_data(inputhandler, frac=0.66,
                                                                                  random_state=random_state,missing_value_strategy=missing_value_strategy)

        monitor = pyltr.models.monitors.ValidationMonitor(
            x_val, y_val, qids_val['q_num'], metric=self.metric, stop_after=250)

        return self.lambdamart.fit(x_train, y_train, qids_train['q_num'], monitor)

    def _predict(self, inputhandler,missing_value_strategy=None):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1,missing_value_strategy=missing_value_strategy)
        print("Predicting...")
        pred = self.lambdamart.predict(x)
        qids = qids.assign(pred=pred)
        tqdm.pandas()
        qids.loc[:, 'rank'] = qids.groupby('q_num')['pred'].progress_apply(pd.Series.rank, ascending=False,
                                                                           method='first')
        qids.drop('pred', inplace=True, axis=1)
        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred


class LambdaMartRandomization(LambdaMart):
    """
    Extends Bonart's LambdaMart wrapper with a predict function for reproducing the results in
    Ferraro, Porcaro, and Serra, ‘Balancing Exposure and Relevance in Academic Search’.
    """

    def __init__(self, featureengineer, sort_reverse=False,random_state=None):
        super().__init__(featureengineer,random_state=random_state)
        self.sort_reverse = sort_reverse

    def __mean_diff(self, relevances):
        return (relevances[-1] - relevances[0]) / len(relevances)

    def __randomize_apply(self, df, randomizer=None):
        if not randomizer:
            randomizer = random.Random()

        df = df.sort_values(by='pred', ascending=not self.sort_reverse)

        pred_list = df.pred.to_list()
        mean_diff = self.__mean_diff(pred_list)
        df.pred = df.apply(lambda row: row.pred + randomizer.uniform(0, mean_diff), axis=1)

        return df

    def _predict(self, inputhandler,missing_value_strategy=None):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1,missing_value_strategy=missing_value_strategy)
        pred = self.lambdamart.predict(x)

        qids = qids.assign(pred=pred)

        tqdm.pandas()
        print("Applying randomization...")
        qids.groupby(['sid', 'q_num']).progress_apply(self.__randomize_apply)

        tqdm.pandas()
        print("Converting relevances to rankings...")
        qids.loc[:, 'rank'] = qids.groupby('q_num')['pred'].progress_apply(pd.Series.rank, ascending=False, method='first')
        qids.drop('pred', inplace=True, axis=1)
        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred
