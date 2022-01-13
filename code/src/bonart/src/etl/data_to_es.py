import glob
import os.path
from pathlib import Path

import jsonlines
from elasticsearch import Elasticsearch
from elasticsearch import helpers

data_path_name = "../m-fair-trec/fair-trec-2019/resources/corpus2019"
data_path = Path(data_path_name)

indexed_corpus_files = "./log/indexed_corpus_files.txt"

if not os.path.exists(indexed_corpus_files):
    with open(indexed_corpus_files, 'w') as pf:
        pass
with open(indexed_corpus_files, "r") as pf:
    processed = pf.read().splitlines()


def doc_generator(reader):
    for doc in reader.iter(type=dict, skip_invalid=True):
        author_names = []
        author_ids = []
        for obj in doc.get('authors'):
            author_ids.extend(obj.get('ids'))
            author_names.append(obj.get('name'))

        yield {
            "_index": 'semanticscholar',
            # "_type": "document", # deprecated
            "_id": doc.get('id'),
            "title": doc.get('title'),
            "paperAbstract": doc.get("paperAbstract"),
            "entities": doc.get("entities"),
            "author_names": author_names,
            "author_ids": author_ids,
            "numInCitations": len(doc.get("inCitations")),
            "numOutCitations": len(doc.get("outCitations")),
            "year": doc.get("year"),
            "venue": doc.get('venue'),
            "journalName": doc.get('journalName'),
            "journalVolume": doc.get('journalVolume'),
            "sources": doc.get('sources'),  # todo: can leave out?
            "doi": doc.get('doi')  # todo: can leave out?
            }


es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

corp_files = glob.glob(data_path_name + "/s2-corpus-[0-9][0-9]")
for corp_file in corp_files:
    # corp_file = data_path_name + "/s2-corpus-42"
    corp_file_name = os.path.basename(corp_file)
    if corp_file_name not in processed:
        print(f"Indexing contents of {corp_file_name}.")
        with jsonlines.open(corp_file) as reader:

            for success, info in helpers.parallel_bulk(es, doc_generator(reader),
                                                       # chunk_size=100, max_chunk_bytes=1000 * 1000 * 25,
                                                       chunk_size=10,
                                                       request_timeout=120):
                if not success:
                    print(f"There was an error: {info}.")
        with open(indexed_corpus_files, "a") as pf:
            pf.write(f"{corp_file_name}\n")

        print(f"Indexed contents of {corp_file}.")
