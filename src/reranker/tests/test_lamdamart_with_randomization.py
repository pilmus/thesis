import os
import random

import numpy as np
import pandas as pd
import pyltr
import pytest
from elasticsearch import Elasticsearch
from tqdm import tqdm

from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMartRandomization


class TestingLambdaMartRandomization(LambdaMartRandomization):
    def __init__(self, sort_reverse=False, random_state=None):
        self.sort_reverse = sort_reverse
        self.random_state = random_state
        pass

    def mean_diff(self, relevances):
        return self._LambdaMartRandomization__mean_diff(relevances)

    def randomize_apply(self, df):
        return self._LambdaMartRandomization__randomize_apply(df)



class TestingLMRPrediction(LambdaMartRandomization):
    def __init__(self,featureengineer,sort_reverse=False):
        super().__init__(featureengineer,random_state=0)
        self.sort_reverse = sort_reverse

    def mean_diff(self, relevances):
        return self._LambdaMartRandomization__mean_diff(relevances)

    def _mean_diffs_apply(self, df):
        df = df.sort_values(by='pred', ascending=not self.sort_reverse)
        preds = df.pred.to_list()
        # pred = df.iloc[0].pred
        md = self.mean_diff(preds)
        df['mean_diff'] = md
        return df

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
        x_val, y_val, qids_val['q_num'], metric=self.metric, stop_after=10)

        return self.lambdamart.fit(x_train, y_train, qids_train['q_num'], monitor)

    def __randomization_apply_q_num_level(self,df):
        id_df = df.groupby('q_num',as_index=False).apply(lambda x: x.reset_index(drop = True)).reset_index(level=1)


    def __randomization_apply(self,df):
        df = df.groupby('q_num').apply()

    def __inspect(self,df):
        md = df.mean_diff.iloc[0]
        if self.random_state is not None:
            rng = random.Random(self.random_state + df.level_0.iloc[0])
        else:
            rng = random.Random()
        df['aug'] = df.apply(lambda row: rng.uniform(0,md),axis=1)
        df['aug_pred'] = df.apply(lambda row: row.pred + row.aug, axis = 1)
        return df

    def predict(self,inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
        pred = self.lambdamart.predict(x)

        qids = qids.assign(pred=pred)
        qids = qids.groupby(['sid', 'q_num'], as_index=False).apply(self._mean_diffs_apply)

        print("Applying randomization...")
        tqdm.pandas()
        qids = qids.groupby(['sid', 'q_num'], as_index=False).apply(lambda x: x.reset_index(drop=True)).reset_index(level=0).groupby(['sid','q_num']).progress_apply(self.__inspect)


        tqdm.pandas()
        print("Converting relevances to rankings...")
        qids["rank"] = qids.groupby(['sid', 'q_num']).aug_pred.progress_apply(pd.Series.rank, method='first', ascending=False)

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred



def mean_diff(rels):
    s = 0
    for i in range(1, len(rels)):
        s += rels[i] - rels[i - 1]
    s = s / len(rels)
    return s

def test_pred():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    qtrain = os.path.join(root,'training','2020','TREC-Fair-Ranking-training-sample.json')
    strain = os.path.join(root,'training','2020','training-sequence-full.tsv')

    qtest = os.path.join(root,'evaluation','2020','TREC-Fair-Ranking-eval-sample.json')
    seqeval = 'seq-test-eval-2020-double-first-double-second.tsv'

    corpus = Corpus('semanticscholar2020og')
    sf = os.path.join(root, 'src/interface/es-features-ferraro-sample-2020.csv')
    ft = FeatureEngineer(corpus, fquery=os.path.join(root, 'config', 'featurequery_ferraro_lmr.json'),
                         fconfig=os.path.join(root, 'config', 'features_ferraro_lmr.json'), feature_mat=sf)

    ioht = InputOutputHandler(corpus,
                             fsequence=strain,
                             fquery=qtrain)
    iohe = InputOutputHandler(corpus,
                             fsequence=seqeval,
                             fquery=qtest)

    lm = TestingLMRPrediction(ft)
    lm.train(ioht)
    lm.predict(iohe)

    lmrev = TestingLMRPrediction(ft, sort_reverse=True)
    lmrev.train(ioht)
    lmrev.predict(iohe)



@pytest.mark.datafiles('/mnt/c/Users/maaik/Documents/thesis/config')
def test_feature_matrix_contains_correct_columns(datafiles):
    config = str(datafiles)
    corpus = Corpus('semanticscholar2020og')
    engineer = FeatureEngineer(corpus, fquery=os.path.join(config, 'featurequery_ferraro_lmr.json'),
                               fconfig=os.path.join(config, 'features_ferraro_lmr.json'))
    features = engineer._FeatureEngineer__get_features('magnetic', ['123cb0363afe113aea915ab5fcdcc75b028a43bd'])
    assert list(features.columns) == ["title_score", "abstract_score", "entities_score", "venue_score", "journal_score",
                                      "authors_score", "inCitations", "outCitations", "doc_id", "qlength"]


def test_mean_diff():
    lm = TestingLambdaMartRandomization()
    np.random.seed(0)
    rels = sorted(list(np.random.uniform(low=0.0, high=1.0, size=(5,))), reverse=True)
    assert lm.mean_diff(rels) == mean_diff(rels)


def test_mean_diff_with_set_list():
    lm = TestingLambdaMartRandomization()
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]
    assert lm.mean_diff(rels) == 0.8 / 5
    assert lm.mean_diff(sorted(rels, reverse=True)) == -0.8 / 5


def test_mean_diff_some_negative_relevances():
    resl = [-0.2, -0.1, 0.3, 0.5, 0.7]
    lm = TestingLambdaMartRandomization()
    assert lm.mean_diff(sorted(resl, reverse=False)) == mean_diff(sorted(resl, reverse=False))
    assert lm.mean_diff(sorted(resl, reverse=True)) == mean_diff(sorted(resl, reverse=True))


def test_mean_diff_all_negative_relevances():
    rels = [-0.8, -0.6, -0.3, -0.05]
    lm = TestingLambdaMartRandomization()
    assert round(lm.mean_diff(sorted(rels, reverse=False)), 6) == round(mean_diff(sorted(rels, reverse=False)), 6)
    assert round(lm.mean_diff(sorted(rels, reverse=True)), 6) == round(mean_diff(sorted(rels, reverse=True)), 6)


def test_apply_randomizer_sort_reverse_false():
    lm = TestingLambdaMartRandomization(random_state=0)
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]
    df = pd.DataFrame({'pred': rels})
    df_out = lm.randomize_apply(df)
    randomizer = random.Random(0)
    check_rels = [rel + randomizer.uniform(0, 0.8 / 5) for rel in rels]
    assert df_out.pred.to_list() == check_rels


def test_apply_randomizer_sort_reverse_true():
    lm = TestingLambdaMartRandomization(sort_reverse=True, random_state=0)
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]
    df = pd.DataFrame({'pred': rels})
    df_out = lm.randomize_apply(df)
    randomizer = random.Random(0)
    check_rels = [rel + randomizer.uniform(0, -0.8 / 5) for rel in sorted(rels, reverse=True)]
    assert df_out.pred.to_list() == check_rels


def test_apply_randomizer_both_ways():
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]

    lm1 = TestingLambdaMartRandomization(sort_reverse=False, random_state=0)
    df1 = pd.DataFrame({'pred': rels})
    df_out1 = lm1.randomize_apply(df1)

    lm2 = TestingLambdaMartRandomization(sort_reverse=True, random_state=0)
    df2 = pd.DataFrame({'pred': rels})
    df_out2 = lm2.randomize_apply(df2)

    assert (df_out1.sort_index() != df_out2.sort_index()).all().all()


def test_apply_randomizer_twice_sort_reverse_true():
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]

    lm1 = TestingLambdaMartRandomization(sort_reverse=True, random_state=0)
    df1 = pd.DataFrame({'pred': rels})
    df_out1 = lm1.randomize_apply(df1)

    lm2 = TestingLambdaMartRandomization(sort_reverse=True, random_state=0)
    df2 = pd.DataFrame({'pred': rels})
    df_out2 = lm2.randomize_apply(df2)

    assert (df_out1.sort_index() == df_out2.sort_index()).all().all()


def test_apply_randomizer_twice_sort_reverse_false():
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]

    lm1 = TestingLambdaMartRandomization(sort_reverse=False, random_state=0)
    df1 = pd.DataFrame({'pred': rels})
    df_out1 = lm1.randomize_apply(df1)

    lm2 = TestingLambdaMartRandomization(sort_reverse=False, random_state=0)
    df2 = pd.DataFrame({'pred': rels})
    df_out2 = lm2.randomize_apply(df2)

    assert (df_out1.sort_index() == df_out2.sort_index()).all().all()


@pytest.mark.datafiles('/mnt/c/Users/maaik/Documents/thesis/config')
def test_train_and_predict_on_same_toy_example_perfect_result(datafiles):
    pass
    # config = str(datafiles)
    # queries = "sample-test.jsonl"
    # sequence = "seq-test.tsv"
    #
    #
    # corpus = Corpus('semanticscholar2020og')
    # ft = FeatureEngineer(corpus, fquery=os.path.join(config,'featurequery_ferraro_lmr.json'),
    #                      fconfig=os.path.join(config,'/features_ferraro_lmr.json'), feature_mat=sf)
    #
    # input_train = InputOutputHandler(corpus,
    #                                  fsequence=sequence,
    #                                  fquery=queries)
    #
    #
    # lambdamart = LambdaMartRandomization(ft, random_state=0)
    # lambdamart.train(ioh)
    # lambdamart.predict(ioh)


def test_predict_reverse_and_not_different_rankings():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    queries = 'sample-test-2020-two-rel-two-not.jsonl'
    seq = 'seq-test-2020.tsv'
    corpus = Corpus('semanticscholar2020og')
    sf = os.path.join(root, 'src/interface/es-features-ferraro-sample-2020.csv')
    ft = FeatureEngineer(corpus, fquery=os.path.join(root, 'config', 'featurequery_ferraro_lmr.json'),
                         fconfig=os.path.join(root, 'config', 'features_ferraro_lmr.json'), feature_mat=sf)

    ioh = InputOutputHandler(corpus,
                                                     fsequence=seq,
                                                     fquery=queries)

    lm = LambdaMartRandomization(ft, random_state=0)
    lm.train(ioh)
    lm.predict(ioh)
    p = lm.predictions

    lmr = LambdaMartRandomization(ft, random_state=0, sort_reverse=True)
    lmr.train(ioh)
    lmr.predict(ioh)
    p_rev = lmr.predictions

    assert (p == p_rev).all().all()

