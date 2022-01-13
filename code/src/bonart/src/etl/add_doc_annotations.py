import os

import pandas as pd
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
annot = '../resources/2020/doc-annotations.csv'
cleaned_training_sample = '../resources/2019/'




def doc_annotation_generator():
    df = pd.read_csv(annot)
    df = df.where(pd.notnull(df), None)


    for row in df.itertuples():
      yield {
        '_op_type': 'update',
        '_index': 'semanticscholar',
        '_type': 'document',
        '_id': row.id,
        'doc': {'advanced': row.Advanced,
                'developing': row.Developing,
                'doclevel':row.DocLevel,
                'h':row.H,
                'l':row.L,
                'doc_hlevel':row.DocHLevel}
        }


def add_doc_annotation_mappings():
    body = {'properties':
                    {'advanced': 'integer',
                     'developing': 'integer',
                     'doclevel': 'keyword',
                     'h': 'integer',
                     'l': 'integer',
                     'doc_hlevel': 'keyword'}}
    es.indices.put_mapping(body=body,
                           index='semanticscholar')


if __name__ == '__main__':
    for success, info in helpers.streaming_bulk(es, doc_annotation_generator(),request_timeout=120,
                                           raise_on_exception=False,raise_on_error=False):
        if not success:
            print(f"There was an error: {info}.")
