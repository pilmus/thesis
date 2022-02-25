import os

import numpy as np
import pytest
from elasticsearch import Elasticsearch

from interface.corpus import Corpus
from interface.features import FeatureEngineer
from reranker.lambdamart import LambdaMartRandomization


class TestingLambdaMartRandomization(LambdaMartRandomization):
    def __init__(self):
        pass

    def mean_diff(self, relevances):
        return self._LambdaMartRandomization__mean_diff(relevances)


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
