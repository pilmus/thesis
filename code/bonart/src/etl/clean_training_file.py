import random

import pandas as pd
from elasticsearch import helpers, Elasticsearch

import src.utils.io as io
from src.interface.corpus import Corpus
from src.interface.iohandler import InputOutputHandler

es = Elasticsearch(timeout=120)


def clean_2019():
    """
    It has come to our attention that the training query set includes document IDs that cannot be resolved in the
    OpenCorpus download.
    We therefore recommend removing such IDs.
    There is a two-step procedure you can follow to remove such documents:

        Remove all documents from query 'documents' sets that do not exist in OpenCorpus
        Drop all queries that, after this document removal, have fewer than 5 documents

    This will result in dropping approximately 100 training queries.
    """

    dirty_training = '../resources/2019/training/fair-TREC-training-sample.json'
    clean_training = '../resources/2019/training/fair-TREC-training-sample-cleaned.json'

    dirty_df = pd.read_json(dirty_training, lines=True)

    dirty_ids = list(set([doc['doc_id'] for doc in dirty_df.documents.explode()]))
    b = {"query": {"ids": {"values": dirty_ids}}}
    res = helpers.scan(es, query=b, index='semanticscholar')

    es_df = pd.DataFrame(res)
    es_ids_set = set(es_df['_id'].to_list())

    clean_df = dirty_df
    clean_df['doc_ids'] = clean_df.apply(lambda row: [doc['doc_id'] for doc in row['documents']], axis=1)

    to_delete = []
    for id, row in clean_df.iterrows():
        row_doc_ids_set = set(row.doc_ids)
        if len(row_doc_ids_set.intersection(es_ids_set)) < len(row_doc_ids_set):
            to_delete.append(id)

    clean_df = clean_df.drop(to_delete)
    clean_df = clean_df.drop('doc_ids', axis=1)
    clean_df.to_json(clean_training, orient='records', lines=True)


def dummy_annot_generator(ids):
    doclevel = ["Advanced", "Developing", "Mixed"]
    doc_hlevel = ["Mixed", 'L', "H"]

    for id in ids:
        yield {
            '_op_type': 'update',
            '_index': 'semanticscholar',
            '_type': 'document',
            '_id': id,
            'doc': {'advanced': random.randint(0, 10),
                    'developing': random.randint(0, 10),
                    'doclevel': random.choice(doclevel),
                    'h': random.randint(0, 10),
                    'l': random.randint(0, 10),
                    'doc_hlevel': random.choice(doc_hlevel)}
            }


def add_dummy_2020_metadata_to_2019():
    """So we can try out DELTR """
    training = '../resources/2019/training/fair-TREC-training-sample-cleaned.json'
    df = pd.read_json(training, lines=True)

    ids = list(set([doc['doc_id'] for doc in df.documents.explode()]))


    for success, info in helpers.streaming_bulk(Elasticsearch(), dummy_annot_generator(ids),
                                                request_timeout=120,
                                                raise_on_exception=False, raise_on_error=False):
        if not success:
            print(f"There was an error: {info}.")




if __name__ == '__main__':
    add_dummy_2020_metadata_to_2019()
