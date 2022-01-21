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

    def __init__(self, host, port,index):
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


    def get_docs_by_ids(self, doc_ids):
        """queries the documents table"""
        s = Search(using=self.es)
        s = s.query("ids", values=doc_ids)
        return self.__return_res_attr_dict_as_df(s)

