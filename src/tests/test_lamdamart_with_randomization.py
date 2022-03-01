import os
import random

import numpy as np
import pandas as pd
import pytest
from elasticsearch import Elasticsearch

from interface.corpus import Corpus
from interface.features import FeatureEngineer
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


def mean_diff(rels):
    s = 0
    for i in range(1, len(rels)):
        s += rels[i] - rels[i - 1]
    s = s / len(rels)
    return s


@pytest.mark.datafiles('../../config')
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


def test_difference_between_reverse_and_not_is_twice_the_random_number():
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]

    df1 = pd.DataFrame({'pred': rels})
    df2 = pd.DataFrame({'pred': rels})

    lm = TestingLambdaMartRandomization(sort_reverse=False, random_state=0)
    lmr = TestingLambdaMartRandomization(sort_reverse=True, random_state=0)

    dfout1 = lm.randomize_apply(df1)
    dfout2 = lmr.randomize_apply(df2)

    mdiff = mean_diff(rels)

    r = random.Random(0)
    rlist = [r.uniform(0, mdiff) for i in range(0, len(rels))]

    lm.mean_diff(sorted(rels, reverse=False))
