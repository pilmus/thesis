import os

import pandas as pd
import pytest

from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.generate_feature_matrix import get_es_features
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMart


def test_test():
    assert 1 == 1


def test_train_and_predict_on_same_toy_example_perfect_result():
    rootpath = '/mnt/c/Users/maaik/Documents/thesis'
    queries = "sample-test-2019.jsonl"
    sequence = "seq-test-2019.tsv"

    sf = os.path.join(rootpath, 'src/interface/es-features-bonart-sample-cleaned-2019.csv')
    corpus = Corpus('semanticscholar2019og')
    ft = FeatureEngineer(corpus, fquery=os.path.join(rootpath, 'config', 'featurequery_bonart.json'),
                         fconfig=os.path.join(rootpath, 'config', 'features_bonart.json'), feature_mat=sf)

    ioh = InputOutputHandler(corpus,
                             fsequence=sequence,
                             fquery=queries)

    lambdamart = LambdaMart(ft, random_state=0)
    lambdamart.train(ioh)
    lambdamart.predict(ioh)
    lambdamart.predictions


def test_train_and_predict_on_same_toy_example_with_one_rel_two_not_rel():
    rootpath = '/mnt/c/Users/maaik/Documents/thesis'
    queries = "sample-test-2019-one-rel-two-unrel.jsonl"
    sequence = "seq-test-2019.tsv"

    sf = os.path.join(rootpath, 'src/interface/es-features-bonart-sample-cleaned-2019.csv')
    corpus = Corpus('semanticscholar2019og')
    ft = FeatureEngineer(corpus, fquery=os.path.join(rootpath, 'config', 'featurequery_bonart.json'),
                         fconfig=os.path.join(rootpath, 'config', 'features_bonart.json'), feature_mat=sf)

    ioh = InputOutputHandler(corpus,
                             fsequence=sequence,
                             fquery=queries)

    lambdamart = LambdaMart(ft, random_state=0)
    lambdamart.train(ioh)
    lambdamart.predict(ioh)

    pr = lambdamart.predictions

    assert pr[(pr.qid == 5842) & (pr.doc_id == '20a11b4f2023dbab4791074c0b86eb0517c79a8d')]['rank'].all() == 1
    assert pr[(pr.qid == 18605) & (pr.doc_id == 'bfddf2c4078b58aefd05b8ba7000aca1338f16d8')]['rank'].all() == 1
    assert pr[(pr.qid == 24207) & (pr.doc_id == '4084f0c34391b3baa4ae2f5b01d7b085ec64f9e5')]['rank'].all() == 1
    assert pr[(pr.qid == 67689) & (pr.doc_id == '26ca17eae88d6f8fd0ff3fc21376dd0b932d6453')]['rank'].all() == 1
    assert pr[(pr.qid == 8081) & (pr.doc_id == '8ffb6ed9de800a43317b63e2e7fa05c2df90db20')]['rank'].all() == 1


def test_prepared_data_same_irrespective_of_using_saved_features_or_newly_retrieved():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    sf = 'es-features-test-2019.csv'
    corp = 'semanticscholar2019og'
    fq = os.path.join(root, 'config', 'featurequery_bonart.json')
    features = os.path.join(root, 'config', 'features_bonart.json')

    seq = 'seq-feature-retrieval.csv'
    queries = os.path.join(root, 'src/interface/full-sample-2020.json')
    get_es_features(corp, fq, features,
                    seq, queries,
                    sf)

    corpus = Corpus(corp)
    ft = FeatureEngineer(corpus, fquery=fq,
                         fconfig=features)
    ioh = InputOutputHandler(corp, seq, queries)

    lm = LambdaMart(ft, random_state=0)

    ftsaved = FeatureEngineer(corpus, fquery=fq,
                              fconfig=features, feature_mat=sf)
    lmsaved = LambdaMart(ftsaved, random_state=0)

    x_train, y_train, qids_train, x_val, y_val, qids_val = lm._prepare_data(ioh)
    x_trains, y_trains, qids_trains, x_vals, y_vals, qids_vals = lmsaved._prepare_data(ioh)

    assert x_train.all().all() == x_trains.all().all()
    assert y_train.all() == y_trains.all()
    assert qids_train.all().all() == qids_trains.all().all()
    assert x_val.all().all() == x_vals.all().all()
    assert y_val.all() == y_vals.all()
    assert qids_val.all().all() == qids_vals.all().all()
