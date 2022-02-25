import time

import jsonlines
import pytest
from elasticsearch import Elasticsearch

# todo: test for having initialized the featurestore
from bonart.interface.corpus import Corpus
from bonart.interface.features import FeatureEngineer
from indexing.data_to_es import index_file, doc_generator


@pytest.fixture
def testfile():
    return './testdoc.jsonl'


@pytest.fixture
def testfile_smaller():
    return './testdoc2.jsonl'


@pytest.fixture
def testidxname():
    return 'testidx'


@pytest.fixture
def testidxname2():
    return 'testidx2'


@pytest.fixture
def es_index1(testidxname):
    es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
    es.indices.create(testidxname)
    yield testidxname
    es.indices.delete(testidxname)


@pytest.fixture
def es_index2(testidxname2):
    es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
    es.indices.create(testidxname2)
    yield testidxname2
    es.indices.delete(testidxname2)


@pytest.fixture
def jsonl_reader(testfile):
    reader = jsonlines.open(testfile)
    return reader


@pytest.fixture
def featureengineer(testidxname):
    return FeatureEngineer(Corpus(index=testidxname), fquery='featurequery_test.json',
                           fconfig='features_test.json')


@pytest.fixture
def featureengineer(testidxname):
    return FeatureEngineer(Corpus(index=testidxname), fquery='featurequery_test.json',
                           fconfig='features_test.json')


@pytest.fixture
def featureengineer_smaller(testidxname2):
    return FeatureEngineer(Corpus(index=testidxname2), fquery='featurequery_test.json',
                           fconfig='features_test.json')


def test_index_created(es_index1):
    es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
    assert es.indices.exists(es_index1)


def test_doc_generator_index_name_updated(jsonl_reader, testidxname):
    docgen = doc_generator(jsonl_reader, 2019, idxname=testidxname)
    dictt = next(docgen)
    print(dictt)
    assert dictt['_index'] == testidxname


def test_compare_bm25_scores_fewer_and_more_fields(es_index1, es_index2, featureengineer,
                                                   featureengineer_smaller):
    index_file('./testdoc.jsonl', 2019, idxname=es_index1)
    index_file('./testdoc2.jsonl', 2019, idxname=es_index2)
    features1 = featureengineer._FeatureEngineer__get_features("test", ["1"])
    features2 = featureengineer_smaller._FeatureEngineer__get_features("test", ["1"])
    assert features1.title_score[0] == features2.title_score[0]

# todo: test that all features are there, num columns == what we expect for example
# todo: test that 2019og, 2020og have correct size (46947044, ...

# todo: test for correct mapping after indexing

# todo: sanchecks: range of dates between 1900 and 2022, numcitations >= 0,


# todo: check that there is a value >0 for each field when the query term is an exact match

# todo: _prepare_data(inputhandler,frac=1) gives same result always
