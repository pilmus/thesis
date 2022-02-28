import pytest
from elasticsearch import Elasticsearch, helpers
from jsonlines import jsonlines
from tqdm import tqdm

from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.iohandler import InputOutputHandler


@pytest.fixture
def index():
    pass


def doc_generator(reader, idxname):
    for doc in reader.iter(type=dict, skip_invalid=True):
        yd = {
            "_index": idxname,
            "_type": "document",
            "_id": doc.get('id'),
            "year": doc.get("year"),
        }

        yield yd


def index_file(raw, idxname):
    print(f"Indexing contents of {raw}.")
    es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
    with jsonlines.open(raw) as reader:
        progress = tqdm(unit="docs", total=1000000)
        successes = 0
        for ok, action in helpers.streaming_bulk(es, doc_generator(reader, idxname=idxname), chunk_size=2000):
            progress.update(1)
            successes += ok


def test_replace_missing_years_with_zero():
    es = Elasticsearch()
    index_file('test_for_missing_year_corpus.jsonl', idxname='test_missing_year_idx')
    corpus = Corpus('test_missing_year_idx')
    fe = FeatureEngineer(corpus, fquery='featurequery_test_missing_year.json',
                         fconfig='features_test_missing_year.json')

    inputhandler = InputOutputHandler(corpus,
                                      fsequence='sequence_missing_year.csv',
                                      fquery='test_for_missing_year_sample.json')

    features = fe.get_feature_mat(inputhandler)
    assert len(features.year) == 4
    assert features.year.iloc[1] == 0
    assert features.year.iloc[3] == 0
    es.indices.delete('test_missing_year_idx')


def test_replace_missing_years_with_average():
    es = Elasticsearch()
    index_file('test_for_missing_year_corpus.jsonl', idxname='test_missing_year_idx')
    corpus = Corpus('test_missing_year_idx')
    fe = FeatureEngineer(corpus, fquery='featurequery_test_missing_year.json',
                         fconfig='features_test_missing_year.json')

    inputhandler = InputOutputHandler(corpus,
                                      fsequence='sequence_missing_year.csv',
                                      fquery='test_for_missing_year_sample.json')

    features = fe.get_feature_mat(inputhandler, missing_value_strategy='avg')
    assert len(features.year) == 4
    assert features.year.iloc[1] == 1011
    assert features.year.iloc[3] == 1011
    es.indices.delete('test_missing_year_idx')


def test_drop_missing_years():
    es = Elasticsearch()
    index_file('test_for_missing_year_corpus.jsonl', idxname='test_missing_year_idx')
    corpus = Corpus('test_missing_year_idx')
    fe = FeatureEngineer(corpus, fquery='featurequery_test_missing_year.json',
                         fconfig='features_test_missing_year.json')

    inputhandler = InputOutputHandler(corpus,
                                      fsequence='sequence_missing_year.csv',
                                      fquery='test_for_missing_year_sample.json')

    features = fe.get_feature_mat(inputhandler, missing_value_strategy='dropzero')
    assert len(features.year) == 2
    assert features.year.iloc[0] == 2022
    assert features.year.iloc[1] == 2022
    es.indices.delete('test_missing_year_idx')

