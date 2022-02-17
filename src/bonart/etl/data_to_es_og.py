from elasticsearch import Elasticsearch
from elasticsearch import helpers
import jsonlines
import re
import pandas as pd
from tqdm import tqdm

def doc_generator(reader):
    for doc in reader.iter(type=dict, skip_invalid=True):
        author_names = []
        author_ids = []
        for obj in doc.get('authors'):
            author_ids.extend(obj.get('ids'))
            author_names.append(obj.get('name'))

        yield {
            "_index": 'testidx',
            "_type": "document",
            "_id": doc.get('id'),
            "title": doc.get('title'),
            "abstract": doc.get("paperAbstract"),
            "entities": doc.get("entities"),
            "author_names": author_names,
            "author_ids": author_ids,
            "num_in_citations": len(doc.get("inCitations")),
            "num_out_citations": len(doc.get("outCitations")),
            "year": doc.get("year"),
            "venue": doc.get('venue'),
            "journalName": doc.get('journalName'),
            "journal_volume": doc.get('journalVolume'),
            "sources": doc.get('sources'),
            "doi": doc.get('doi')
        }


es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])


file = '/mnt/d/tempdir/sample-S2-records'
with jsonlines.open(file) as reader:
    progress = tqdm(unit="docs", total=1000000)
    successes = 0
    for ok, action in helpers.streaming_bulk(es, doc_generator(reader), chunk_size=2000):
        progress.update(1)
        successes += ok