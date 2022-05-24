import pickle
import random

import numpy as np
import pyltr
import pandas as pd
from tqdm import tqdm

from app.pre_processing.pre_processor import get_preprocessor
from app.pre_processing.src.iohandler import IOHandler
from app.reranking.src import model

from sklearn.calibration import IsotonicRegression

from app.reranking.src.post_process_reranker import MRFR


class LambdaMart(model.RankerInterface):
    """
    Wrapper around the LambdaMart algorithm
    """

    def __str__(self):
        return f"LM_{self.random_state}_{self.early_stopping_frac}_{self._metric_name}"

    def __init__(self, random_state=None, early_stopping_frac=0.6, metric='NDCG', save_dir=None, feature_numbers=None):
        super().__init__()
        self.fe = get_preprocessor().fe
        self._metric_name = metric
        if metric == 'NDCG':
            self.metric = pyltr.metrics.NDCG(k=7)
        elif metric == 'ERR':
            self.metric = pyltr.metrics.ERR(1)
        else:
            raise ValueError("Invalid metric: ", metric)

        self.early_stopping_frac = early_stopping_frac

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
        self.save_dir = save_dir
        self.train_ioh = None
        self.feature_numbers = feature_numbers

    def __data_helper(self, x):
        x = x.sort_values('q_num')
        qids = x[['sid', 'q_num', 'doc_id']]
        y = x['relevance']
        x.drop(['q_num', 'doc_id', 'relevance', 'qid'], inplace=True, axis=1)
        return (x, y, qids)

    def _prepare_data(self, inputhandler, frac):
        if frac < 1:
            impute = True
        else:
            impute = False  # todo: unjank this, should not let imputation depend on whether we use a frac or not. make it a class variable instead that is set to true when training and false when evaluating https://stats.stackexchange.com/a/425086
        x = self.fe.get_feature_mat(inputhandler, compute_impute=impute, feature_numbers=self.feature_numbers)
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

        self.train_ioh = inputhandler

        x_train, y_train, qids_train, x_val, y_val, qids_val = self._prepare_data(inputhandler,
                                                                                  frac=self.early_stopping_frac)

        monitor = pyltr.models.monitors.ValidationMonitor(
            x_val, y_val, qids_val['q_num'], metric=self.metric, stop_after=250)

        self.lambdamart.fit(x_train, y_train, qids_train['q_num'], monitor)

        if self.save_dir:
            self.save()

    def _predict(self, inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
        print("Predicting...")
        pred = self.lambdamart.predict(x)
        predictions = self._base_pred(inputhandler, pred, qids)
        return predictions

    def _base_pred(self, inputhandler, pred, qids):
        qids = qids.assign(pred=pred)
        tqdm.pandas()
        qids.loc[:, 'rank'] = qids.groupby('q_num')['pred'].progress_apply(pd.Series.rank, ascending=False,
                                                                           method='first')
        qids.drop('pred', inplace=True, axis=1)
        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred

    def save(self, fitted):
        savename = f"{self.__class__.__name__}_{self.random_state}_{self.train_ioh}"
        with open(savename, 'wb') as handle:
            pickle.dump(self.lambdamart, handle)

    def load(self):
        pass


class LambdaMartMRFR(LambdaMart):
    def __init__(self, random_state, early_stopping_frac=0.6, metric='NDCG', feature_numbers=None, ranker_config=None):
        super().__init__(random_state, early_stopping_frac=early_stopping_frac, metric=metric, feature_numbers=feature_numbers)
        self.ranker_config = ranker_config

    def _predict(self, inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
        print("Predicting...")
        pred = self.lambdamart.predict(x)
        predictions = self._base_pred(inputhandler, pred, qids)
        self._pred_to_est_rel(inputhandler, pred, qids)

        return predictions

    def _pred_to_est_rel(self, inputhandler, pred, qids):
        qids = qids.assign(est_relevance=pred)
        predictions = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids)
        predictions['est_relevance'] = predictions.groupby('qid')['est_relevance'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        predictions = predictions.sort_values(by=['sid', 'q_num', 'est_relevance'])
        predictions[['qid', 'doc_id', 'est_relevance']].to_csv(
            f'reranking/resources/relevances/{self.ranker_config}.csv', index=False)  # todo: un-hardcode
        return predictions


class LambdaMartRandomization(LambdaMart):
    """
    Extends Bonart's LambdaMart wrapper with a predict function for reproducing the results in
    Ferraro, Porcaro, and Serra, ‘Balancing Exposure and Relevance in Academic Search’.
    """

    def __init__(self, sort_reverse=False, random_state=None):
        super().__init__(random_state=random_state)
        self.sort_reverse = sort_reverse
        self.mean_diff_dict = {}

    def __mean_diff(self, relevances):
        return (relevances[-1] - relevances[0]) / len(relevances)

    def __randomize_apply(self, df):
        df = df.sort_values(by='pred', ascending=not self.sort_reverse)
        sid = df.sid.iloc[0]
        if sid not in self.mean_diff_dict:
            self.mean_diff_dict[sid] = self.__mean_diff(df.pred.to_list())
        md = self.mean_diff_dict[sid]

        if self.random_state is not None:
            rng = random.Random(self.random_state + df.q_num.iloc[0])
        else:
            rng = random.Random()

        df['aug'] = df.apply(lambda row: rng.uniform(0, md), axis=1)
        df['aug_pred'] = df.apply(lambda row: row.pred + row.aug, axis=1)
        return df

    def _predict(self, inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
        print("Predicting values...")
        tqdm.pandas()
        pred = self.lambdamart.predict(x)

        qids = qids.assign(pred=pred)

        print("Applying randomization...")
        tqdm.pandas()
        qids = qids.groupby(['sid', 'q_num']).progress_apply(self.__randomize_apply)

        tqdm.pandas()
        print("Converting relevances to rankings...")
        qids = qids.reset_index(drop=True)
        qids["rank"] = qids.groupby(['sid', 'q_num']).aug_pred.progress_apply(pd.Series.rank, method='first',
                                                                              ascending=False)

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred
