import json

import pandas as pd
from tqdm import tqdm


class FeatureEngineer:
    """Returns feature vectors for provided query-doc_ids pairs"""

    def __init__(self, corpus, fquery, fconfig, feature_mat=None, missing_value_strategy='avg', init_ltr=True):
        if corpus:
            self.corpus = corpus

            with open(fquery) as f:
                dat = json.load(f)
            self.query = dat

            if init_ltr:
                corpus.init_ltr(fconfig)
        self.feature_mat = None
        if feature_mat:
            self.feature_mat = feature_mat
        if not feature_mat and not corpus:
            raise ValueError(
                f"You must either initialize a corpus, fquery, and fconfig or give a pre-generated feature matrix!")
        self.missing_value_strategy = missing_value_strategy
        self.missing_values = None

    def _get_features(self, queryterm, doc_ids):
        #todo: queryterm is >=1 words?
        docs = self._get_es_ltr_resp(doc_ids, queryterm)
        resp = self._features_from_response(docs)
        resp = self._add_query_features(queryterm, resp)
        return resp

    def _get_es_ltr_resp(self, doc_ids, queryterm):
        self.query['query']['bool']['filter'][0]['terms']['_id'] = doc_ids
        self.query['query']['bool']['filter'][1]['sltr']['params']['keywords'] = queryterm
        docs = self.corpus.es.search(index=self.corpus.index, body=self.query, size=len(doc_ids))
        return docs

    def _add_query_features(self, queryterm, resp):
        resp['qlength'] = len(queryterm)
        return resp

    @property
    def log_field_name(self):
        return self.query["ext"]["ltr_log"]["log_specs"]["name"]

    def get_feature_mat(self, iohandler, *args, **kwargs):
        print("Getting features...")
        if self.feature_mat:
            f = pd.read_csv(self.feature_mat, dtype={'doc_id': object, 'qid': str})
            qs = iohandler.get_query_seq()[['qid', 'doc_id']].drop_duplicates()
            feature_mat = pd.merge(f, qs, on=['qid', 'doc_id'], how='right')
            fm = feature_mat
        else:
            tqdm.pandas()
            features = iohandler.get_query_seq().groupby('qid').progress_apply(
                lambda df: self._get_features(df['query'].iloc[0], df['doc_id'].unique().tolist()))

            features = features.reset_index(
                level=0)  # brings the qid back as a column after having been used to groupby

            fm = features
        if self.missing_value_strategy == 'dropzero':  # todo: move this to the feature engineer?
            fm = fm.dropna()
            fm = fm[fm.year != 0]
        elif self.missing_value_strategy == 'avg':
            if not self.missing_values:
                # this method is first encountered when training. we then want to set the "missing value" to the
                # mean of the training set. when we second encounter this method, we don't change the method, but use
                # the mean of the training set to impute the test set as well. https://stats.stackexchange.com/a/301353
                self.missing_values = self._impute_means(fm)
            for col in fm.columns.to_list():
                if col == 'doc_id':
                    continue
                fm[col] = fm[col].fillna(self.missing_values[col])
            fm.year = fm.year.replace(0, self.missing_values['year'])

        else:
            raise ValueError(f"Invalid missing value strategy: {self.missing_value_strategy}")
        return fm

    def _features_from_response(self, docs):
        docs = docs['hits']['hits']
        features = [doc['fields']['_ltrlog'][0][self.log_field_name] for doc in docs]
        ids = [doc['_id'] for doc in docs]
        result = []
        for i, vec in enumerate(features):
            vec = {el['name']: el['value'] for el in vec}
            vec['doc_id'] = ids[i]
            result.append(vec)
        return pd.DataFrame.from_dict(result)

    def _impute_means(self, x):  # todo: test
        missing_values = {}
        for col in x.columns.to_list():
            if col == 'doc_id':
                continue
            # df[~df['Age'].isna()]
            if col == 'year':
                missing_values['year'] = x[(x.year != 0) & ~x.year.isna()].year.mean() #nans are not counted in means
                continue
            missing_values[col] = x[~x[col].isna()][col].mean()

        return missing_values


class AnnotationFeatureEngineer(FeatureEngineer):
    def __init__(self, doc_annotations, corpus=None, fquery=None, fconfig=None, init_ltr=True, es_feature_mat=None):
        super(AnnotationFeatureEngineer, self).__init__(corpus, fquery, fconfig, init_ltr, es_feature_mat)
        self.doc_annotations = pd.read_csv(doc_annotations).rename({'id': 'doc_id'}, axis=1)

    def get_feature_mat(self, iohandler, annotation_features=None, *args, **kwargs):
        if not annotation_features:
            annotation_features = ['doc_id']
        else:
            annotation_features = ['doc_id'] + annotation_features
        es_features = super().get_feature_mat(iohandler)
        doc_annotations = self.doc_annotations[annotation_features]
        features = pd.merge(es_features, doc_annotations, on='doc_id', how='left')
        features = features.dropna()
        return features


class ExtendedFeatureEngineer(FeatureEngineer):
    """Adds a number of features based on other sources than an elasticsearch ltr featureset."""

    FIELDS = ["title","paperAbstract", "venue", "journalName", "author_names","sources","fields_of_study"]

    def _get_features(self, queryterm, doc_ids):
        featdf = super()._get_features(queryterm,doc_ids)
        esdf = self.__get_es_resp(doc_ids)
        featdf = pd.merge(featdf,esdf,on='doc_id')
        return featdf

    def __get_es_resp(self, doc_ids):
        query = {'query':{'ids':{'values':doc_ids}}}
        resp = self.corpus.es.search(index=self.corpus.index, body=query, size=len(doc_ids))
        hits = resp['hits']['hits']
        records = []
        for hit in hits:
            resdict = {}
            resdict['doc_id'] = hit['_id']
            src = hit['_source']
            for field in self.FIELDS:
                if type(src[field]) == list:
                    resdict[f'{field}length_char'] = len(" ".join(src[field]))
                    resdict[f'{field}length_token'] = len(" ".join(src[field]).split())
                else:
                    resdict[f'{field}length_char'] = len(src[field])
                    resdict[f'{field}length_token'] = len(src[field].split())
            records.append(resdict)
        return pd.DataFrame(records)

    def _add_query_features(self, queryterm, resp):
        resp['qlength_char'] = len(queryterm)
        resp['qlength_token'] = len(queryterm.split())
        return resp