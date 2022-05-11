import os

import pandas as pd
from sklearn.datasets import dump_svmlight_file

from app.pre_processing.src.corpus import Corpus
from app.pre_processing.src.features import FeatureEngineer, ExtendedFeatureEngineer
from app.pre_processing.src.iohandler import IOHandler

PREPROCESSOR = None


def get_preprocessor():
    global PREPROCESSOR
    if not PREPROCESSOR:
        PREPROCESSOR = PreProcessor()
    return PREPROCESSOR


class PreProcessor():
    _app_entry = None
    _corpus = None
    _fe = None
    _ioht = None
    _iohe = None

    def __init__(self):
        print("ape")

    def init(self, app_entry, extend=None):
        self._app_entry = app_entry
        preproc_config = app_entry.preproc_config

        preproc_components = {
            "_corpus": Corpus,
            "_fe": FeatureEngineer,
            "_ioht": IOHandler,
            "_iohe": IOHandler
        }

        if extend:  # todo; ew, fix this
            preproc_components["_fe"] = ExtendedFeatureEngineer

        for k, v in preproc_config.items():
            component_class = preproc_components[k]
            component_params = []
            for val in v.values():
                component_params.append(eval(val))
            setattr(self, k, component_class(*component_params))

    def get_feature_mat(self):
        # either return feature matrix FILE or return None
        rn = self._app_entry.reranker_name
        esf_path = os.path.join('pre_processing', 'resources', 'escache', f'{rn}.csv')
        if os.path.exists(esf_path):
            return esf_path
        return None

    def save_feature_mat(self, fm):
        rn = self._app_entry.reranker_name
        esf_path = os.path.join('pre_processing', 'resources', 'escache', f'{rn}.csv')
        fm.to_csv(esf_path, index=False)

    def dump_svm(self, X, y, qids):
        rn = self._app_entry.reranker_name
        svm_path = os.path.join('pre_processing', 'resources', 'svmcache', f'{rn}.csv')
        dump_svmlight_file(X, y, svm_path, query_id=qids)

    @property
    def fe(self):
        return self._fe

    @fe.setter
    def fe(self, value):
        self._fe = value

    @property
    def ioht(self):
        return self._ioht

    @property
    def iohe(self):
        return self._iohe
