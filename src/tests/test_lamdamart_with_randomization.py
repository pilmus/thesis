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
    def __init__(self,sort_reverse = False):
        self.sort_reverse = sort_reverse
        pass

    def mean_diff(self, relevances):
        return self._LambdaMartRandomization__mean_diff(relevances)

    def randomize_apply(self, df, randomizer):
        return self._LambdaMartRandomization__randomize_apply(df, randomizer)


def mean_diff(rels):
    return np.convolve(rels, np.array([1, -1]), 'valid').sum() / len(rels)


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
    rels = [0.1,0.2,0.5,0.7,0.9]
    assert lm.mean_diff(rels) == 0.8/5
    assert lm.mean_diff(sorted(rels,reverse=True)) == -0.8/5

def test_apply_randomizer_sort_reverse_false():
    lm = TestingLambdaMartRandomization()
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]
    df = pd.DataFrame({'pred': rels})
    df_out = lm.randomize_apply(df,random.Random(0))
    randomizer = random.Random(0)
    check_rels = [rel + randomizer.uniform(0,0.8/5) for rel in rels]
    assert df_out.pred.to_list() == check_rels

def test_apply_randomizer_sort_reverse_true():
    lm = TestingLambdaMartRandomization(sort_reverse=True)
    rels = [0.1, 0.2, 0.5, 0.7, 0.9]
    df = pd.DataFrame({'pred': rels})
    df_out = lm.randomize_apply(df,random.Random(0))
    randomizer = random.Random(0)
    check_rels = [rel + randomizer.uniform(0,-0.8/5) for rel in sorted(rels,reverse=True)]
    assert df_out.pred.to_list() == check_rels

