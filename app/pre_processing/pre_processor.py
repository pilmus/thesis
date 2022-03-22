import os

from app.pre_processing.src.corpus import Corpus
from app.pre_processing.src.features import FeatureEngineer
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

    def init(self, app_entry):
        self._app_entry = app_entry
        preproc_config = app_entry.preproc_config

        preproc_components = {
        "_corpus" : Corpus,
        "_fe" : FeatureEngineer,
        "_ioht" : IOHandler,
        "_iohe" : IOHandler
        }


        for k,v in preproc_config.items():
            component_class = preproc_components[k]
            component_params = []
            for val in v.values():
                print(val)
                component_params.append(eval(val))
            setattr(self,k, component_class(*component_params))

        #
        #
        # self._corpus = Corpus(app_entry.get_argument('index'))
        #
        # try:
        #     self._fe = FeatureEngineer(self._corpus,
        #                            app_entry.get_argument('fquery'),
        #                            app_entry.get_argument('fconfig'),
        #                            self.get_feature_mat())
        # except Exception:
        #     pass
        #
        # try:
        #     self._ioht = IOHandler(app_entry.get_argument('fsequence_train'),
        #                        app_entry.get_argument('fquery_train'))
        #
        # except Exception:
        #     pass
        #
        # self._iohe = IOHandler(app_entry.get_argument('fsequence_eval'),
        #                        app_entry.get_argument('fquery_eval'))

    def get_feature_mat(self):
        # either return feature matrix FILE or return None
        rn = self._app_entry.reranker_name
        esf_path = os.path.join('pre_processing', 'resources', 'escache', f'{rn}.csv')
        if os.path.exists(esf_path):
            return esf_path
        return None

    def get_query_seq(self):
        pass

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
