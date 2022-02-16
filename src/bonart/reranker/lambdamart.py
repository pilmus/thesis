import pickle
import random

import pyltr
import pandas as pd
import src.bonart.reranker.model as model


class LambdaMart(model.RankerInterface):
    """
    Wrapper around the LambdaMart algorithm
    """

    def __init__(self, featureengineer):
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
            verbose=1)

    def __data_helper(self, x):
        x = x.sort_values('q_num')
        qids = x[['sid', 'q_num', 'doc_id']]
        y = x['relevance']
        x.drop(['q_num', 'doc_id', 'relevance', 'qid'], inplace=True, axis=1)
        return (x, y, qids)

    def _prepare_data(self, inputhandler, frac=0.66, prepped_data=None):
        if prepped_data:
            x = pd.read_csv(prepped_data)
        else:
            x = self.fe.get_feature_mat_from_iohandler(inputhandler)
        y = inputhandler.get_query_seq()[['sid', 'qid', "q_num", "doc_id", "relevance"]]
        x = pd.merge(x, y, how="left", on=['qid', 'doc_id'])
        training = x.q_num.drop_duplicates().sample(frac=frac)
        x_train, y_train, qids_train = self.__data_helper(x.loc[x.q_num.isin(training)])

        x_val, y_val, qids_val = [None] * 3

        if frac < 1:
            x_val, y_val, qids_val = self.__data_helper(x.loc[~x.q_num.isin(training)])

        return (x_train, y_train, qids_train, x_val, y_val, qids_val)

    def train(self, inputhandler, prepped_data=None):
        """
        X : array_like, shape = [n_samples, n_features] Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
        y : array_like, shape = [n_samples] Target values (integers in classification, real numbers in regression)
        For classification, labels must correspond to classes.
        qids : array_like, shape = [n_samples] Query ids for each sample. Samples must be grouped by query such that
        all queries with the same qid appear in one contiguous block.
        """

        x_train, y_train, qids_train, x_val, y_val, qids_val = self._prepare_data(inputhandler, frac=0.66,
                                                                                  prepped_data=prepped_data)

        monitor = pyltr.models.monitors.ValidationMonitor(
            x_val, y_val, qids_val['q_num'], metric=self.metric, stop_after=250)

        return self.lambdamart.fit(x_train, y_train, qids_train['q_num'], monitor)

    def _predict(self, inputhandler, prepped_data=None):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1, prepped_data=prepped_data)
        pred = self.lambdamart.predict(x)
        qids = qids.assign(pred=pred)
        qids.loc[:, 'rank'] = qids.groupby('q_num')['pred'].apply(pd.Series.rank, ascending=False, method='first')
        qids.drop('pred', inplace=True, axis=1)
        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred

    def save(self, path):
        print(f"Saving models...")

        with open(path, 'wb') as fp:
            pickle.dump(self.lambdamart, fp)
        return True

    def load(self, path):
        with open(path, "rb") as fp:
            self.lambdamart = pickle.load(fp)


class LambdaMartFerraro(LambdaMart):
    """
    Extends Bonart's LambdaMart wrapper with a predict function for reproducing the results in
    Ferraro, Porcaro, and Serra, ‘Balancing Exposure and Relevance in Academic Search’.
    """

    def __init__(self, featureengineer, sort_reverse=False):
        super().__init__(featureengineer)
        self.sort_reverse = sort_reverse

    def __randomize_apply(self, df):
        pred_list = sorted(df.pred.to_list(), reverse=self.sort_reverse)
        mean_diff = (pred_list[-1] - pred_list[0]) / len(pred_list)
        df.pred = df.apply(lambda row: row.pred + random.uniform(0, mean_diff), axis=1)

        return df

    def _predict(self, inputhandler, prepped_data=None):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1, prepped_data=prepped_data)
        pred = self.lambdamart.predict(x)

        qids = qids.assign(pred=pred)

        qids.groupby(['sid', 'q_num']).apply(self.__randomize_apply)

        qids.loc[:, 'rank'] = qids.groupby('q_num')['pred'].apply(pd.Series.rank, ascending=False, method='first')
        qids.drop('pred', inplace=True, axis=1)
        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred
