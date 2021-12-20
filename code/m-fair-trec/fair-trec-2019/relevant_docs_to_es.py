# based on code by Malte Bonart

import gzip
import re
from pathlib import Path

import jsonlines
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers

data_path = "resources/corpus2019"

processed_files = "./processed_zips.txt"

with open(processed_files, "r") as pf:
    processed = pf.read().splitlines()


def doc_generator(reader, reldocids):
    for doc in reader.iter(type=dict, skip_invalid=True):
        if doc['id'] in reldocids:
            author_names = []
            author_ids = []
            for obj in doc.get('authors'):
                author_ids.extend(obj.get('ids'))
                author_names.append(obj.get('name'))

            yield {
                "_index": 'semanticscholar',
                "_type": "document",
                "_id": doc.get('id'),
                "title": doc.get('title'),
                "paperAbstract": doc.get("paperAbstract"),
                "entities": doc.get("entities"),
                "author_names": author_names,
                "author_ids": author_ids,
                "inCitations": len(doc.get("inCitations")),
                "outCitations": len(doc.get("outCitations")),
                "year": doc.get("year"),
                "venue": doc.get('venue'),
                "journalName": doc.get('journalName'),
                "journalVolume": doc.get('journalVolume'),
                "sources": doc.get('sources'),
                "doi": doc.get('doi')
                }


es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

path = Path(data_path)


reldocids = 'fair-TREC-docids.txt'
with open(reldocids, 'r') as f:
    reldocids = set(f.read().splitlines())


for file in path.iterdir():
    if file.suffix == ".gz":
        if file.name not in processed:
            print(f'Processing {file.name}.')
            with gzip.open(str(file)) as f:
                reader = jsonlines.Reader(f)
                gen = doc_generator(reader, reldocids)
                for doc in gen:
                    print(doc)
                for success, info in helpers.parallel_bulk(es, doc_generator(reader),
                                                           chunk_size=100, max_chunk_bytes=1000 * 1000 * 25,
                                                           request_timeout=30):
                    if not success:
                        print(f"Doc {file.name} failed.")
            with open(processed_files, "a") as pf:
                pf.write(f"{file.name}\n")

            print(f"Processed {file.name}.")
