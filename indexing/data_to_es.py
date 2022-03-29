import glob
import os.path

from elasticsearch import helpers, Elasticsearch
import tqdm

import jsonlines

logdir = 'resources/logs'


def already_indexed(indexed_files):
    if not os.path.exists(indexed_files):
        with open(indexed_files, "w"):
            pass
    with open(indexed_files, "r") as fp:
        return fp.read().splitlines()


def doc_generator(reader, year, idxname=None):
    for doc in reader.iter(type=dict, skip_invalid=True):
        author_names = []
        author_ids = []
        for obj in doc.get('authors'):
            author_ids.extend(obj.get('ids'))
            author_names.append(obj.get('name'))

        yd = {
            "_index": 'semanticscholar2019og',
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

        if year == 2019:
            pass
        elif year == 2020:
            yd["sources"] = doc.get("sources")
            yd["fields_of_study"] = doc.get("fieldsOfStudy")
            yd["_index"] = "semanticscholar2020og"

        else:
            raise ValueError(f"Invalid year: {year}.")

        if idxname:
            yd["_index"] = idxname

        yield yd


def update_generator(reader, field, year, idxname=None):
    for doc in reader.iter(type=dict, skip_invalid=True):

        yd = {"_op_type": "update",
              "_index": 'semanticscholar2019og',
              "_id": doc.get('id'),
              "_source":{'doc':{field: doc.get(field)}}
              }

        if year == 2019:
            pass
        elif year == 2020:
            yd["_index"] = "semanticscholar2020og"

        else:
            raise ValueError(f"Invalid year: {year}.")

        if idxname:
            yd["_index"] = idxname

        yield yd


def index_files(year):
    rawspath = f'/mnt/d/corpus{year}/'
    raw_files = glob.glob(rawspath + "*")

    print(f"Raw files: {raw_files}.")

    indexed_filepath = os.path.join(logdir, f'indexed_files_{year}.txt')
    indexed_files = already_indexed(indexed_filepath)

    print(f"Already indexed: {indexed_files}.")
    for raw in raw_files:
        if raw not in indexed_files:
            index_file(raw, year)

            with open(indexed_filepath, "a") as fp:
                fp.write(f"{raw}\n")
                print(f"Indexed contents of {raw}.")


def index_file(raw, year, idxname=None):
    print(f"Indexing contents of {raw}.")
    es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
    with jsonlines.open(raw) as reader:
        progress = tqdm.tqdm(unit="docs", total=1000000)
        successes = 0
        for ok, action in helpers.streaming_bulk(es, doc_generator(reader, year, idxname=idxname), chunk_size=2000):
            progress.update(1)
            successes += ok


def update_file(raw, field, year, idxname=None):
    print(f"Updating contents of {raw}.")
    es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
    with jsonlines.open(raw) as reader:
        progress = tqdm.tqdm(unit="docs", total=1000000)
        successes = 0
        for ok, action in helpers.streaming_bulk(es, update_generator(reader, field, year, idxname=idxname),
                                                 chunk_size=2000):
            progress.update(1)
            successes += ok


def update_files():
    pass
