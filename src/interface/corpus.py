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

    def __init__(self, index, host='localhost', port='9200'):
        """needs a path to a valid .json db configuration file"""
        self.es = Elasticsearch([{'host': host, 'port': port, 'timeout': 300}])
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
        s = Search(index=self.index, using=self.es)
        resp = s.query().count()
        return resp

    def __res_to_df(self, res):
        ids = [d.meta.id for d in res]
        df = pd.DataFrame((d.to_dict() for d in res))
        df['doc_id'] = ids
        log.info("fetched %s doc_ids", df.doc_id.nunique())
        return df

    def get_docs_by_ids(self, doc_ids):
        """queries the documents table"""
        s = Search(index=self.index, using=self.es)
        s = s.query("ids", values=doc_ids)
        if len(doc_ids) > 10000:
            return self.__res_to_df(s.scan())
        else:
            return self.__res_to_df(s[:len(doc_ids)].execute())
