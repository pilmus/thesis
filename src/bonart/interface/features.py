import json
import sys

import pandas as pd
from tqdm import tqdm

import src.bonart.utils.io as io
from src.bonart.interface.iohandler import Queries


class FeatureEngineer:
    """Returns feature vectors for provided query-doc_ids pairs"""

    def __init__(self, corpus, fquery, fconfig,
                 query_df: pd.DataFrame,
                 init_ltr=True,
                 qlen_is_feature=False,
                 metadata: str = None,
                 grouping: str = None):
        self.corpus = corpus
        self.query = io.read_json(fquery)
        self.query_df = query_df

        self._features = None

        self.qlen_is_feature = qlen_is_feature
        self.metadata = metadata
        self.grouping = grouping

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
    def features(self):
        if not self._features:
            raise SystemError("You need to get some features first!")

    @features.setter
    def features(self, value):
        self._features(value)

    # def get_feature_mat_from_iohandler(self, iohandler):
    #     print("Getting features...")
    #
    #     features = self._get_feature_mat(iohandler.get_query_seq())
    #     return features
    #
    # def get_feature_mat_from_queries(self, queries: Queries) -> pd.DataFrame:
    #     print("Getting features from queries...")
    #     tqdm.pandas()
    #     features = self._get_feature_mat(queries.queries)
    #     return features

    def get_feature_mat(self):
        es_features = self._get_es_features()
        metadata_features = self._get_metadata_features()
        qlen_feature = self._get_qlens()
        group_feature = self._apply_grouping()

        features = pd.merge(es_features, metadata_features,  on='qid')
        features = pd.merge(features, qlen_feature, on='qid')
        features = pd.merge(features,group_feature, on='doc_id')
        self.features = features

        #todo what happens if there are no es features?

    # @property
    # def grouping(self):
    #     return self._grouping
    #
    # @grouping.setter
    # def grouping(self, value):
    #     self._grouping = pd.read_csv(value)

    def _apply_grouping(self):
        if self.grouping:
            grouping = pd.read_csv(self.grouping)
            df = self.query_df[['qid','doc_id']]
            df = df.apply(pd.merge(df, grouping, how='left', on='doc_id'))
            return df
        else:
            return pd.DataFrame(columns=['doc_id'])


    def _get_qlens(self):
        if self.qlen_is_feature:
            df = self.query_df[['qid', 'query']]
            df['qlen'] = df.query.apply(lambda q: len(q))
            return df[['qid', 'qlen']]
        else:
            return pd.DataFrame(columns=['qid'])

    def _get_metadata_features(self):
        if self.metadata:
            return pd.read_csv(self.metadata)
        else:
            return pd.DataFrame(columns=['qid'])

    def _get_es_features(self):
        tqdm.pandas()
        features = self.query_df.groupby('qid').progress_apply(
            lambda df: self.__get_es_features(df['query'].iloc[0], df['doc_id'].unique().tolist()))
        features = features.reset_index(level=0)
        self.features = features
        return features

    def __get_es_features(self, queryterm, doc_ids):
        self.query['query']['bool']['filter'][0]['terms']['_id'] = doc_ids
        self.query['query']['bool']['filter'][1]['sltr']['params']['keywords'] = queryterm
        docs = self.corpus.es.search(index=self.corpus.index, body=self.query, size=len(doc_ids))
        resp = self.__features_from_es_response(docs)
        # resp['qlength'] = len(queryterm) #todo: move out of here
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
