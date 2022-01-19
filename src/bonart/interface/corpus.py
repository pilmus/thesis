import json
import logging as log

import pandas as pd
import requests
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q
from pandas.io.json import json_normalize

import src.bonart.utils.io as io


class Corpus():
    """Interface between the corpus data on Elasticsearch and other modules. All fetched data is returned as a pandas
    dataframe."""

    def __init__(self, host='localhost', port='9200', index='semanticscholar'):
        """needs a path to a valid .json db configuration file"""
        self.es = Elasticsearch([{'host': host, 'port': port, 'timeout':120}])
        self.index = index
        self.host = host
        self.port = port

    def init_ltr(self, fconfig):
        feature_set = io.read_json(fconfig)
        base = 'http://' + self.host + ":" + str(self.port) + '/'
        requests.put(base + '_ltr')
        full_path = base + "_ltr/_featureset/" + feature_set['featureset']['name']
        head = {'Content-Type': 'application/json'}
        resp = requests.post(full_path, data=json.dumps(feature_set), headers=head)
        return resp

    def count_docs(self):
        s = Search(using=self.es)
        resp = s.query().count()
        return resp

    def __return_res_attr_dict_as_df(self, s):
        ids = [d.meta.id for d in s.scan()]
        df = pd.DataFrame((d.to_dict() for d in s.scan()))
        df['doc_id'] = ids
        log.info("fetched %s doc_ids", df.doc_id.nunique())
        return df

    def __return_res_dict_as_df(self, s):
        df = pd.DataFrame(s)
        df = pd.concat([df, df["_source"].apply(pd.Series)],axis = 1)
        df['doc_id'] = df['_id']
        return df

    def get_docs_by_authors(self, author_ids):
        """queries the author table"""
        s = Search(using=self.es)
        s = s.filter('terms', author_ids=author_ids)
        return self.__return_res_attr_dict_as_df(s)

    def get_docs_by_ids(self, doc_ids):
        """queries the documents table"""
        s = Search(using=self.es)
        s = s.query("ids", values=doc_ids)
        return self.__return_res_attr_dict_as_df(s)

    def get_docs_with_2020_metadata(self, doc_ids=None):
        """
        Get all docs that have the metadata that was provided for trec 2020.

        :param doc_ids: filter by the ids in the list if param present
        :return:
        """
        if doc_ids:

            body = {"query": {"bool": {"must": [{"exists": {"field": "l"}}, {"ids": {"values": doc_ids}}]}}}


        else:
            body = {"query": {"exists": {"field": "l"}}}
        res = helpers.scan(Elasticsearch(), query=body, index='semanticscholar')
        return self.__return_res_dict_as_df(res)

    # def get_authors(self, doc_ids):
    #     docs = self.get_docs_by_ids(doc_ids)
    #     docs = docs[["doc_id", "author_ids"]]
    #     docs = docs.to_dict(orient='records')
    #     docs = json_normalize(docs, 'author_ids', 'doc_id')
    #     docs.rename(columns={0: 'author_id'}, inplace=True)
    #     log.info("fetched %s author ids from %s of %s provided doc_ids", docs.author_id.nunique(),
    #              docs.doc_id.nunique(), len(doc_ids))
    #     return docs
