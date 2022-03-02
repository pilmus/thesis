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
        self.random_state = random_state

    def __data_helper(self, x):
        x = x.sort_values('q_num')
        qids = x[['sid', 'q_num', 'doc_id']]
        y = x['relevance']
        x.drop(['q_num', 'doc_id', 'relevance', 'qid'], inplace=True, axis=1)
        return (x, y, qids)

    def _get_feature_mat(self, inputhandler):
        return self.fe.get_feature_mat(inputhandler)

    def _prepare_data(self, inputhandler, frac=0.66):
        x = self._get_feature_mat(inputhandler)
        y = inputhandler.get_query_seq()[['sid', 'qid', "q_num", "doc_id", "relevance"]]
        x = pd.merge(x, y, how="left", on=['qid', 'doc_id'])
        training = x.q_num.drop_duplicates().sample(frac=frac, random_state=self.random_state)
        x_train, y_train, qids_train = self.__data_helper(x.loc[x.q_num.isin(training)])

        x_val, y_val, qids_val = [None] * 3

        if frac < 1:
            x_val, y_val, qids_val = self.__data_helper(x.loc[~x.q_num.isin(training)])

        return (x_train, y_train, qids_train, x_val, y_val, qids_val)

    def train(self, inputhandler):
        """
        X : array_like, shape = [n_samples, n_features] Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
        y : array_like, shape = [n_samples] Target values (integers in classification, real numbers in regression)
        For classification, labels must correspond to classes.
        qids : array_like, shape = [n_samples] Query ids for each sample. Samples must be grouped by query such that
        all queries with the same qid appear in one contiguous block.
        """

        x_train, y_train, qids_train, x_val, y_val, qids_val = self._prepare_data(inputhandler, frac=0.66)

        monitor = pyltr.models.monitors.ValidationMonitor(
        x_val, y_val, qids_val['q_num'], metric=self.metric, stop_after=250)

        return self.lambdamart.fit(x_train, y_train, qids_train['q_num'], monitor)

    def _predict(self, inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
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


class LambdaMartYear(LambdaMart):
    def __init__(self, featureengineer, random_state, missing_value_strategy):
        super().__init__(featureengineer, random_state)
        self.missing_value_strategy = missing_value_strategy
        self.missing_value = None

    def _get_feature_mat(self, inputhandler):
        x = self.fe.get_feature_mat(inputhandler)
        if self.missing_value_strategy == 'dropzero':
            x = x[x.year != 0]
        elif self.missing_value_strategy == 'avg':
            if not self.missing_value:
                # this method is first encountered when training. we then want to set the "missing value" to the
                # mean of the training set. when we second encounter this method, we don't change the method, but use
                # the mean of the training set to impute the test set as well. https://stats.stackexchange.com/a/301353
                self.missing_value = self._impute_mean(x)
            x.year = x.year.replace(0, self.missing_value)
        elif not self.missing_value:
            pass
        else:
            raise ValueError(f"Invalid missing value strategy: {self.missing_value_strategy}")

        return x

    def _impute_mean(self, x):  # todo: test
        return x[x.year != 0].year.mean()


class LambdaMartRandomization(LambdaMart):
    """
    Extends Bonart's LambdaMart wrapper with a predict function for reproducing the results in
    Ferraro, Porcaro, and Serra, ‘Balancing Exposure and Relevance in Academic Search’.
    """

    def __init__(self, featureengineer, sort_reverse=False, random_state=None):
        super().__init__(featureengineer, random_state=random_state)
        self.sort_reverse = sort_reverse

    def __mean_diff(self, relevances):
        return (relevances[-1] - relevances[0]) / len(relevances)

    def __randomizer(self, row, addition):
        return row + addition

    def __randomize_apply(self, df):
        randomizer = random.Random(self.random_state)
        df = df.sort_values(by='pred', ascending=not self.sort_reverse)

        pred_list = df.pred.to_list()
        mean_diff = self.__mean_diff(pred_list)
        df.pred = df.pred.apply(lambda row: self.__randomizer(row, randomizer.uniform(0, mean_diff)))

        return df

    def _predict(self, inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
        pred = self.lambdamart.predict(x)

        qids = qids.assign(pred=pred)

        tqdm.pandas()
        print("Applying randomization...")
        qids = qids.groupby(['sid', 'q_num'], as_index=False).progress_apply(self.__randomize_apply)

        tqdm.pandas()
        print("Converting relevances to rankings...")
        qids.loc[:, 'rank'] = qids.groupby('sid')['pred'].progress_apply(pd.Series.rank, ascending=False,
                                                                         method='first')

        qids.drop('pred', inplace=True, axis=1)
        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred
