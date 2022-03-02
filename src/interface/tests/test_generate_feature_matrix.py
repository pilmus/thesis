import os

import pandas as pd

from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.generate_feature_matrix import get_es_features
from interface.iohandler import InputOutputHandler

import pytest

def test_generated_matrix_equal_to_retrieved_matrix():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    sf = 'es-features-test-2019.csv'
    corp = 'semanticscholar2019og'
    fq = os.path.join(root,'config','featurequery_bonart.json')
    features = os.path.join(root,'config','features_bonart.json')

    seq = 'seq-feature-retrieval.csv'
    queries = os.path.join(root, 'src/interface/full-sample-2020.json')
    get_es_features(corp, fq, features,
                    seq, queries,
                    sf)

    corpus = Corpus(corp)
    ft = FeatureEngineer(corpus, fquery=fq,
                         fconfig=features)
    ioh = InputOutputHandler(corp,seq,queries)
    retr_features = ft.get_feature_mat(ioh)

    assert (pd.read_csv(sf,float_precision='high') == retr_features.reset_index(drop=True)).all().all()



