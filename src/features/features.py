import pandas as pd
from tqdm import tqdm

import utils.io as io


class FeatureEngineer:
    """Returns feature vectors for provided query-doc_ids pairs"""

    def __init__(self, corpus, fquery, fconfig, init_ltr=True, feature_mat=None):
        if corpus:
            self.corpus = corpus
            self.query = io.read_json(fquery)
            if init_ltr:
                corpus.init_ltr(fconfig)
        if feature_mat:
            self.feature_mat = feature_mat
        if not feature_mat and not corpus:
            raise ValueError(
                f"You must either initialize a corpus, fquery, and fconfig or give a pre-generated feature matrix!")

    def __get_features(self, queryterm, doc_ids):
        self.query['query']['bool']['filter'][0]['terms']['_id'] = doc_ids
        self.query['query']['bool']['filter'][1]['sltr']['params']['keywords'] = queryterm
        docs = self.corpus.es.search(index=self.corpus.index, body=self.query, size=len(doc_ids))
        resp = self.__features_from_response(docs)
        resp['qlength'] = len(queryterm)
        return resp

    @property
    def log_field_name(self):
        return self.query["ext"]["ltr_log"]["log_specs"]["name"]

    def get_feature_mat(self, iohandler, *args, **kwargs):
        print("Getting features...")
        if self.feature_mat:
            f = pd.read_csv(self.feature_mat, dtype={'doc_id': object})
            qs = iohandler.get_query_seq()[['qid', 'doc_id']].drop_duplicates()
            feature_mat = pd.merge(f, qs, on=['qid', 'doc_id'])
            return feature_mat
        else:
            tqdm.pandas()
            features = iohandler.get_query_seq().groupby('qid').progress_apply(
                lambda df: self.__get_features(df['query'].iloc[0], df['doc_id'].unique().tolist()))

            features = features.reset_index(
                level=0)  # brings the qid back as a column after having been used to groupby

            return features

    def __features_from_response(self, docs):
        docs = docs['hits']['hits']
        features = [doc['fields']['_ltrlog'][0][self.log_field_name] for doc in docs]
        ids = [doc['_id'] for doc in docs]
        result = []
        for i, vec in enumerate(features):
            vec = {el['name']: el['value'] for el in vec}
            vec['doc_id'] = ids[i]
            result.append(vec)
        return pd.DataFrame.from_dict(result)


class AnnotationFeatureEngineer(FeatureEngineer):
    def __init__(self, doc_annotations, corpus=None, fquery=None, fconfig=None, init_ltr=True, es_feature_mat=None):
        super(AnnotationFeatureEngineer, self).__init__(corpus, fquery, fconfig, init_ltr, es_feature_mat)
        self.doc_annotations = pd.read_csv(doc_annotations).rename({'id': 'doc_id'}, axis=1)

    def get_feature_mat(self, iohandler, annotation_features=None,*args,**kwargs):
        if not annotation_features:
            annotation_features = ['doc_id']
        else:
            annotation_features = ['doc_id'] + annotation_features
        es_features = super().get_feature_mat(iohandler)
        doc_annotations = self.doc_annotations[annotation_features]
        features = pd.merge(es_features, doc_annotations, on='doc_id', how='left')
        features = features.dropna()
        return features
