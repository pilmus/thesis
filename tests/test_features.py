import os

import pytest
from elasticsearch import Elasticsearch, helpers
from jsonlines import jsonlines
from tqdm import tqdm

from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.generate_feature_matrix import get_es_features
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMartYear


@pytest.fixture
def index():
    pass


def doc_generator_year(reader, idxname):
    for doc in reader.iter(type=dict, skip_invalid=True):
        yd = {
            "_index": idxname,
            "_type": "document",
            "_id": doc.get('id'),
            "year": doc.get("year"),
        }

        yield yd


def doc_generator_title_numcitations(reader, idxname):
    for doc in reader.iter(type=dict, skip_invalid=True):
        yd = {
            "_index": idxname,
            "_type": "document",
            "_id": doc.get('id'),
            "title": doc.get("title"),
            "inCitations": len(doc.get("inCitations"))
        }

        yield yd


def index_file(raw, idxname, generator):
    print(f"Indexing contents of {raw}.")
    es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
    with jsonlines.open(raw) as reader:
        progress = tqdm(unit="docs", total=1000000)
        successes = 0
        for ok, action in helpers.streaming_bulk(es, generator(reader, idxname=idxname), chunk_size=2000):
            progress.update(1)
            successes += ok


def test_replace_missing_years_with_zero():
    es = Elasticsearch()
    index_file('corpus_test_for_missing_year.jsonl', idxname='test_missing_year_idx', generator=doc_generator_year)
    corpus = Corpus('test_missing_year_idx')
    fe = FeatureEngineer(corpus, fquery='featurequery_test_missing_year.json',
                         fconfig='features_test_missing_year.json')

    inputhandler = InputOutputHandler(corpus,
                                      fsequence='seq_missing_year.csv',
                                      fquery='sample_test_for_missing_year.json')

    lm = LambdaMartYear(fe, random_state=None, missing_value_strategy=None)
    features = lm._get_feature_mat(inputhandler)
    assert len(features.year) == 4
    assert features.year.iloc[1] == 0
    assert features.year.iloc[3] == 0
    es.indices.delete('test_missing_year_idx')


def test_impute_missing_years_with_mean():
    es = Elasticsearch()
    index_file('corpus_test_for_missing_year.jsonl', idxname='test_missing_year_idx', generator=doc_generator_year)
    corpus = Corpus('test_missing_year_idx')
    fe = FeatureEngineer(corpus, fquery='featurequery_test_missing_year.json',
                         fconfig='features_test_missing_year.json')

    inputhandler = InputOutputHandler(corpus,
                                      fsequence='seq_missing_year.csv',
                                      fquery='sample_test_for_missing_year.json')

    lm = LambdaMartYear(fe, random_state=None, missing_value_strategy='avg')
    features = lm._get_feature_mat(inputhandler)

    assert len(features.year) == 4
    assert features.year.iloc[1] == 2022
    assert features.year.iloc[3] == 2022
    es.indices.delete('test_missing_year_idx')


def test_drop_missing_years():
    es = Elasticsearch()
    index_file('corpus_test_for_missing_year.jsonl', idxname='test_missing_year_idx', generator=doc_generator_year)
    corpus = Corpus('test_missing_year_idx')
    fe = FeatureEngineer(corpus, fquery='featurequery_test_missing_year.json',
                         fconfig='features_test_missing_year.json')

    inputhandler = InputOutputHandler(corpus,
                                      fsequence='seq_missing_year.csv',
                                      fquery='sample_test_for_missing_year.json')

    lm = LambdaMartYear(fe, random_state=None, missing_value_strategy='dropzero')
    features = lm._get_feature_mat(inputhandler)
    assert len(features.year) == 2
    assert features.year.iloc[0] == 2022
    assert features.year.iloc[1] == 2022
    es.indices.delete('test_missing_year_idx')


def test_pre_generated_feature_matrix_equivalent_to_on_the_fly_feature_matrix():
    es = Elasticsearch()
    corp = 'test_equiv_features_idx'
    index_file('corpus_equiv_features.jsonl', idxname=('%s' % corp),
               generator=doc_generator_title_numcitations)

    fquery = 'featurequery_equiv_feature_gen.json'
    fconfig = 'features_equiv_feature_gen.json'
    seq = 'seq_equiv_features.csv'
    queries = 'sample_equiv_features.jsonl'
    saved_features = 'es-features_equiv_features.csv'

    get_es_features(corp, fquery, fconfig,
                    seq, queries,
                    saved_features)

    corpus = Corpus(corp)
    ft_otf = FeatureEngineer(corpus, fquery=fquery,
                             fconfig=fconfig)
    ioh = InputOutputHandler(corpus,
                             fsequence=seq,
                             fquery=queries)
    on_the_fly_features = ft_otf.get_feature_mat(ioh)

    ft_preload = FeatureEngineer(corpus, fquery=fquery, fconfig=fconfig, feature_mat=saved_features)

    preload_features = ft_preload.get_feature_mat(ioh)

    assert set(on_the_fly_features.columns) == set(preload_features.columns)
    assert (on_the_fly_features.reset_index(drop=True) == preload_features[on_the_fly_features.columns].reset_index(
        drop=True)).all().all()

    es.indices.delete(corp)


def test_more_entries_in_sample_than_in_sequence_should_be_limited_to_seq_only():
    es = Elasticsearch()
    corp = 'test_equiv_features_idx'
    index_file('corpus_equiv_features.jsonl', idxname=corp,
               generator=doc_generator_title_numcitations)

    fquery = 'featurequery_equiv_feature_gen.json'
    fconfig = 'features_equiv_feature_gen.json'
    longerseq = 'seq_more_in_sample.csv'
    seq = 'seq_equiv_features.csv'
    queries = 'sample_more_entries_than_in_sequence.jsonl'
    saved_features = 'es-features_moresample_features.csv'

    get_es_features(corp, fquery, fconfig,
                    longerseq, queries,
                    saved_features)

    corpus = Corpus(corp)
    ft_otf = FeatureEngineer(corpus, fquery=fquery,
                             fconfig=fconfig)
    ioh = InputOutputHandler(corpus,
                             fsequence=seq,
                             fquery=queries)
    on_the_fly_features = ft_otf.get_feature_mat(ioh)

    ft_preload = FeatureEngineer(corpus, fquery=fquery, fconfig=fconfig, feature_mat=saved_features)

    preload_features = ft_preload.get_feature_mat(ioh)

    assert set(on_the_fly_features.columns) == set(preload_features.columns)
    assert (on_the_fly_features.reset_index(drop=True) == preload_features[on_the_fly_features.columns].reset_index(
        drop=True)).all().all()

    os.remove(saved_features)
    es.indices.delete(corp)


@pytest.mark.datafiles('../../')
def test_same_output_for_large_featureset_semanticscholar2019og(datafiles):
    root = str(datafiles)
    corp = 'semanticscholar2019og'

    fquery = os.path.join(root,'config/featurequery_bonart.json')
    fconfig = os.path.join(root,'config/features_bonart.json')

    seq_train = os.path.join(root,'training/2019/training-sequence-full.tsv')
    queries_train = os.path.join(root,'training/2019/fair-TREC-training-sample-cleaned.json')

    seq_test = os.path.join(root,'evaluation/2019/fair-TREC-evaluation-sequences.csv')
    queries_test = os.path.join(root,'evaluation/2019/fair-TREC-evaluation-sample.json')

    saved_features = os.path.join(root,'src/interface/es-features-bonart-sample-cleaned-2019.csv')

    corpus = Corpus(corp)
    ft_otf = FeatureEngineer(corpus, fquery=fquery,
                             fconfig=fconfig)
    ioh_train = InputOutputHandler(corpus,
                                   fsequence=seq_train,
                                   fquery=queries_train)
    ioh_test = InputOutputHandler(corpus,
                                  fsequence=seq_test,
                                  fquery=queries_test)

    on_the_fly_features_train = ft_otf.get_feature_mat(ioh_train)
    on_the_fly_features_test = ft_otf.get_feature_mat(ioh_test)

    ft_preload = FeatureEngineer(corpus, fquery=fquery, fconfig=fconfig, feature_mat=saved_features)
    preload_features_train = ft_preload.get_feature_mat(ioh_train)
    preload_features_test = ft_preload.get_feature_mat(ioh_test)

    assert set(on_the_fly_features_train.columns) == set(preload_features_train.columns)

    assert set(on_the_fly_features_test.columns) == set(preload_features_test.columns)

    assert (on_the_fly_features_train.reset_index(drop=True) == preload_features_train[
        on_the_fly_features_train.columns].reset_index(
        drop=True)).all().all()

    assert (on_the_fly_features_test.reset_index(drop=True) == preload_features_test[
        on_the_fly_features_test.columns].reset_index(
        drop=True)).all().all()

@pytest.mark.datafiles('../../')
def test_same_output_for_large_featureset_semanticscholar2019og(datafiles):
    root = str(datafiles)
    corp = 'semanticscholar2019og'

    fquery = os.path.join(root,'config/featurequery_bonart.json')
    fconfig = os.path.join(root,'config/features_bonart.json')

    seq_train = os.path.join(root,'training/2019/training-sequence-full.tsv')
    queries_train = os.path.join(root,'training/2019/fair-TREC-training-sample-cleaned.json')

    seq_test = os.path.join(root,'evaluation/2019/fair-TREC-evaluation-sequences.csv')
    queries_test = os.path.join(root,'evaluation/2019/fair-TREC-evaluation-sample.json')

    saved_features = os.path.join(root,'src/interface/es-features-bonart-sample-cleaned-2019.csv')

    corpus = Corpus(corp)
    ft_otf = FeatureEngineer(corpus, fquery=fquery,
                             fconfig=fconfig)
    ioh_train = InputOutputHandler(corpus,
                                   fsequence=seq_train,
                                   fquery=queries_train)
    ioh_test = InputOutputHandler(corpus,
                                  fsequence=seq_test,
                                  fquery=queries_test)

    on_the_fly_features_train = ft_otf.get_feature_mat(ioh_train)
    on_the_fly_features_test = ft_otf.get_feature_mat(ioh_test)

    ft_preload = FeatureEngineer(corpus, fquery=fquery, fconfig=fconfig, feature_mat=saved_features)
    preload_features_train = ft_preload.get_feature_mat(ioh_train)
    preload_features_test = ft_preload.get_feature_mat(ioh_test)

    assert set(on_the_fly_features_train.columns) == set(preload_features_train.columns)

    assert set(on_the_fly_features_test.columns) == set(preload_features_test.columns)

    assert (on_the_fly_features_train.reset_index(drop=True) == preload_features_train[
        on_the_fly_features_train.columns].reset_index(
        drop=True)).all().all()

    assert (on_the_fly_features_test.reset_index(drop=True) == preload_features_test[
        on_the_fly_features_test.columns].reset_index(
        drop=True)).all().all()
