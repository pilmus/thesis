import glob
import os.path

from elasticsearch import helpers
import tqdm

import jsonlines
from elasticsearch import Elasticsearch

logdir = 'resources/logs'
es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])

def already_indexed(indexed_files):
    if not os.path.exists(indexed_files):
        with open(indexed_files, "w"):
            pass
    with open(indexed_files, "r") as fp:
        return fp.read().splitlines()


def get_mapping(year):
    base_mapping = {
        "properties": {
            "title": {"type": "text"},
            "abstract": {"type": "text"},
            "entities": {"type": "text"},
            "venue": {"type": "text"},
            "journalName": {"type": "text"},
            "author_names": {"type": "text"},
            "author_ids": {"type": "text"},
            "num_in_citations": {"type": "integer"},
            "num_out_citations": {"type": "integer"},
        }
    }

    if year == 2019:
        base_mapping["properties"]["year"] = {"type": "short"}

    elif year == 2020 or year == '2020subset':
        base_mapping["properties"]["sources"] = {"type": "text"}
        base_mapping["properties"]["fields_of_study"] = {"type": "text"}
    return base_mapping




def doc_generator(reader, year):
    for doc in reader.iter(type=dict, skip_invalid=True):
        author_names = []
        author_ids = []
        for obj in doc.get('authors'):
            author_ids.extend(obj.get('ids'))
            author_names.append(obj.get('name'))

        yield_dict = {

            "_id": doc.get('id'),
            "title": doc.get('title'),
            "abstract": doc.get("paperAbstract"),
            "entities": doc.get("entities"),
            "author_names": author_names,
            "author_ids": author_ids,
            "num_in_citations": len(doc.get("inCitations")),
            "num_out_citations": len(doc.get("outCitations")),
            "venue": doc.get('venue'),
            "journalName": doc.get('journalName'),
        }

        if year == 2019:
            yield_dict["_index"] = 'semanticscholar2019'
            yield_dict['year'] = doc.get("year")

        elif year == 'test':
            yield_dict["_index"] = 'testidx2'
            yield_dict['year'] = doc.get("year")

        elif year == 2020:
            yield_dict["_index"] = 'semanticscholar2020'

            yield_dict["sources"] = doc.get("sources")
            yield_dict["sources_text"] = doc.get('sources'),

            yield_dict["fields_of_study"] = doc.get("fieldsOfStudy")
            yield_dict["fields_of_study_text"] = doc.get('fieldsOfStudy')

        elif year == '2020subset':
            yield_dict["_index"] = 'semanticscholar2020subset'
            yield_dict["sources"] = doc.get("sources")
            yield_dict["fields_of_study"] = doc.get("fieldsOfStudy")
        else:
            print(f"Invalid year {year}.")

        yield yield_dict


def update_doc_generator(reader):
    for doc in reader.iter(type=dict, skip_invalid=True):
        yield_dict = {
            "_op_type": 'update',
            "_index": 'semanticscholar2020',
            "_id": doc.get('id'),
            "doc": {
                "sources_text": doc.get('sources'),
                "fields_of_study_text": doc.get('fieldsOfStudy')}
        }

        yield yield_dict


def new_index(year):
    idx_name = f"semanticscholar{year}"
    es.indices.create(idx_name)
    es.indices.put_mapping(get_mapping(year), index=idx_name)


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

def update_files():
    rawspath = f'/mnt/d/corpus2020/'
    raw_files = glob.glob(rawspath + "*")

    print(f"Raw files: {raw_files}.")

    indexed_filepath = os.path.join(logdir, f'updated_files_2020.txt')
    indexed_files = already_indexed(indexed_filepath)

    print(f"Already indexed: {indexed_files}.")
    for raw in raw_files:
        if raw not in indexed_files:
            update_file(raw)

            with open(indexed_filepath, "a") as fp:
                fp.write(f"{raw}\n")
                print(f"Indexed contents of {raw}.")

def update_file(raw):
    print(f"Updating contents of {raw}.")
    with jsonlines.open(raw) as reader:
        progress = tqdm.tqdm(unit="docs", total=1000000)
        successes = 0
        for ok, action in helpers.streaming_bulk(es, update_doc_generator(reader), chunk_size=3000):
            progress.update(1)
            successes += ok

def index_file(raw, year):
    print(f"Indexing contents of {raw}.")
    with jsonlines.open(raw) as reader:
        progress = tqdm.tqdm(unit="docs", total=1000000)
        successes = 0
        for ok, action in helpers.streaming_bulk(es, doc_generator(reader, year), chunk_size=2000):
            progress.update(1)
            successes += ok


if __name__ == '__main__':

    # update_files()
    update_file('/mnt/d/corpus2020/tempsource')
    print("I'm done.")
