import glob
import os.path

from elasticsearch.helpers import streaming_bulk
import tqdm

import jsonlines
from elasticsearch import Elasticsearch
from elasticsearch import helpers

logdir = 'resources/logs'


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
                "author_ids": {"type": "keyword"},
                "num_in_citations": {"type": "integer"},
                "num_out_citations": {"type": "integer"},
                }
            }

    if year == 2019:
        base_mapping["properties"]["year"] = {"type":"short"}

    elif year == 2020:
        base_mapping["properties"]["sources"] = {"type": "keyword"}
        base_mapping["properties"]["fields_of_study"] = {"type": "keyword"}
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
            "venue": doc.get('venue'),
            "journalName": doc.get('journalName'),
            "author_names": author_names,
            "author_ids": author_ids,

            "num_in_citations": len(doc.get("inCitations")),
            "num_out_citations": len(doc.get("outCitations")),
            }

        if year == 2019:
            yield_dict["_index"] = 'semanticscholar2019'
            yield_dict['year'] = doc.get("year")

        elif year == 2020:
            yield_dict["_index"] = 'semanticscholar2020'
            yield_dict["sources"] = doc.get("sources")
            yield_dict["fields_of_study"] = doc.get("fieldsOfStudy")
        else:
            print(f"Invalid year {year}.")

        yield yield_dict


def index_files(year):
    rawspath = f'resources/corpus/{year}/raw/'
    raw_files = glob.glob(rawspath + "*")

    print(f"Raw files: {raw_files}.")

    indexed_filepath = os.path.join(logdir, f'indexed_files_{year}.txt')
    indexed_files = already_indexed(indexed_filepath)

    print(f"Already indexed: {indexed_files}.")
    for raw in raw_files:
        if raw not in indexed_files:
            print(f"Indexing contents of {raw}.")
            with jsonlines.open(raw) as reader:

                progress = tqdm.tqdm(unit="docs", total=1000000)
                successes = 0
                for ok, action in streaming_bulk(es, doc_generator(reader, year), chunk_size=100):
                    progress.update(1)
                    successes += ok

                #
                #
                #
                # for success, info in helpers.parallel_bulk(es, doc_generator(reader, year), chunk_size=10,
                #                                            request_timeout=120):
                #     if not success:
                #         print(f"There was an error: {info}.")
            with open(indexed_filepath, "a") as fp:
                fp.write(f"{raw}\n")
                print(f"Indexed contents of {raw}.")


es = Elasticsearch([{'host': 'localhost', 'port': '9200', 'timeout': 300}])
index_files(2020)
print("I'm done.")
