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

    def __init__(self, featureengineer, random_state=None, save_dir=None):
        super().__init__()
        self.fe = get_preprocessor().fe
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
        self.save_dir = save_dir
        self.train_ioh = None

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

        self.train_ioh = inputhandler

        x_train, y_train, qids_train, x_val, y_val, qids_val = self._prepare_data(inputhandler, frac=0.66)

        monitor = pyltr.models.monitors.ValidationMonitor(
            x_val, y_val, qids_val['q_num'], metric=self.metric, stop_after=250)

        self.lambdamart.fit(x_train, y_train, qids_train['q_num'], monitor)

        if self.save_dir:
            self.save()



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

    def save(self,fitted):
        savename = f"{self.__class__.__name__ }_{self.random_state}_{self.train_ioh}"
        with open(savename, 'wb') as handle:
            pickle.dump(self.lambdamart, handle)

    def load(self):
        pass


class LambdaMartYear(LambdaMart):
    def __init__(self, featureengineer, random_state, missing_value_strategy):
        super().__init__(featureengineer, random_state)
        self.missing_value_strategy = missing_value_strategy
        self.missing_values = None

    def _get_feature_mat(self, inputhandler):
        x = self.fe.get_feature_mat(inputhandler)
        if self.missing_value_strategy == 'dropzero':
            x = x.dropna()
            x = x[x.year != 0]
        elif self.missing_value_strategy == 'avg':
            if not self.missing_values:
                # this method is first encountered when training. we then want to set the "missing value" to the
                # mean of the training set. when we second encounter this method, we don't change the method, but use
                # the mean of the training set to impute the test set as well. https://stats.stackexchange.com/a/301353
                self.missing_values = self._impute_means(x)
            for col in x.columns.to_list():
                if col == 'doc_id':
                    continue
                x[col] = x[col].fillna(self.missing_values[col])
            x.year = x.year.replace(0, self.missing_values['year'])

        else:
            raise ValueError(f"Invalid missing value strategy: {self.missing_value_strategy}")

        return x

    def _impute_means(self, x):  # todo: test
        missing_values = {}
        for col in x.columns.to_list():
            if col == 'doc_id':
                continue
            # df[~df['Age'].isna()]
            missing_values[col] = x[~x[col].isna()][col].mean()
        # return x[x.year != 0].year.mean()
        return missing_values


class LambdaMartMRFR(LambdaMartYear):
    def __init__(self, featureengineer, random_state, missing_value_strategy,relevance_probabilities,grouping,K,beta,lambd):
        super().__init__(featureengineer, random_state, missing_value_strategy)
        self.relevance_probabilities = relevance_probabilities
        self.grouping = grouping
        self.K = K
        self.beta = beta
        self._lambda = lambd




    def _predict(self, inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
        print("Predicting...")
        pred = self.lambdamart.predict(x)
        qids = qids.assign(est_relevance=pred)
        est_rels = pd.merge(inputhandler.get_query_seq()[['sid', 'qid','doc_id']].drop_duplicates(),qids)[['qid','doc_id','est_relevance']].drop_duplicates()

        # normalize relevance
        # est_rels['est_relevance'] = est_rels.groupby('qid')['est_relevance'].transform(lambda x: (x - x.mean()) / x.std())
        est_rels['est_relevance'] = est_rels.groupby('qid')['est_relevance'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        est_rels = est_rels.sort_values(by=['qid', 'est_relevance'])

        est_rels.to_csv(self.relevance_probabilities,index=False)

        mrfr = MRFR(self.relevance_probabilities,self.grouping,self.K,self.beta,self._lambda)
        outdf = mrfr.rerank(inputhandler)

        predictions = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], outdf, how='left',
                        on=['sid', 'q_num', 'doc_id'])

        # tqdm.pandas()
        # qids.loc[:, 'rank'] = qids.groupby(['sid','q_num'])['pred'].progress_apply(pd.Series.rank, ascending=False,
        #                                                                    method='first')
        # pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
        #                 how='left', on=['sid', 'q_num', 'doc_id'])
        # qids.drop('pred', inplace=True, axis=1)



        return predictions


class LambdaMartRandomization(LambdaMart):
    """
    Extends Bonart's LambdaMart wrapper with a predict function for reproducing the results in
    Ferraro, Porcaro, and Serra, ‘Balancing Exposure and Relevance in Academic Search’.
    """

    def __init__(self, featureengineer, sort_reverse=False, random_state=None):
        super().__init__(featureengineer, random_state=random_state)
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
