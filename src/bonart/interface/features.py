import json
import sys

import pandas as pd
from tqdm import tqdm

import src.bonart.utils.io as io


class FeatureEngineer:
    """Returns feature vectors for provided query-doc_ids pairs"""

    def __init__(self, corpus, fquery, fconfig,
                 query_df: pd.DataFrame,
                 init_ltr=True):
        self.corpus = corpus
        self.query = io.read_json(fquery)
        self.query_df = query_df


        if init_ltr:
            corpus.init_ltr(fconfig)

    @staticmethod
    def load_feature_file(filepath):
        return pd.read_csv(filepath)

    @staticmethod
    def save_features_libsvm(df, path):
        raise NotImplementedError

    @staticmethod
    def save_features_csv(df, path):
        raise NotImplementedError

    @property
    def log_field_name(self):
        return self.query["ext"]["ltr_log"]["log_specs"]["name"]

    @property
    def save_name(self):
        return self.log_field_name.split('_')[1]

    @staticmethod
    def get_feature_mat(query_df, es_fpath=None, metadata_fpath=None, qlen_isfeature=False, group_fpath=None):
        if qlen_isfeature:
            query_df['qlen'] = query_df.query.apply(lambda row: len(row))
            query_df = query_df[['qid', 'doc_id', 'qlen']]
        else:
            query_df = query_df[['qid', 'doc_id']]
        es_features = pd.read_csv(es_fpath) if es_fpath else pd.DataFrame(columns=['qid'])
        metadata_features = pd.read_csv(metadata_fpath).rename({'paper_sha': 'doc_id'}, axis=1) if metadata_fpath else pd.DataFrame(columns=['doc_id'])
        group_feature = pd.read_csv(group_fpath) if group_fpath else pd.DataFrame(columns=['doc_id'])

        features = pd.merge(query_df, es_features, on=['qid','doc_id'], how='left')
        features = pd.merge(features, metadata_features, on='doc_id', how='left')
        features = pd.merge(features, group_feature, on='doc_id', how='left')

        return features

    def _get_es_features(self):
        tqdm.pandas()
        es_features = self.query_df.groupby('qid').progress_apply(
            lambda df: self.__get_es_features(df['query'].iloc[0], df['doc_id'].unique().tolist()))
        es_features = es_features.reset_index(level=0)
        self._es_features = es_features

        es_features.to_csv(f'resources/features/es_features_{self.save_name}.csv',index=False)
        return es_features

    def __get_es_features(self, queryterm, doc_ids):
        self.query['query']['bool']['filter'][0]['terms']['_id'] = doc_ids
        self.query['query']['bool']['filter'][1]['sltr']['params']['keywords'] = queryterm
        docs = self.corpus.es.search(index=self.corpus.index, body=self.query, size=len(doc_ids))
        resp = self.__features_from_es_response(docs)
        return resp

    def __features_from_es_response(self, docs):
        docs = docs['hits']['hits']
        features = [doc['fields']['_ltrlog'][0][self.log_field_name] for doc in docs]
        ids = [doc['_id'] for doc in docs]
        result = []
        for i, vec in enumerate(features):
            vec = {el['name']: el['value'] for el in vec}
            vec['doc_id'] = ids[i]
            result.append(vec)
        return pd.DataFrame.from_dict(result)
