import csv
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

    def init(self, app_entry):
        self._app_entry = app_entry
        preproc_config = app_entry.preproc_config

        preproc_components = {
            "_corpus": Corpus,
            "_fe": FeatureEngineer,
            "_ioht": IOHandler,
            "_iohe": IOHandler
        }

        if preproc_config.pop("extend", None):  # todo; ew, fix this
            preproc_components["_fe"] = ExtendedFeatureEngineer

        for k, v in preproc_config.items():
            component_class = preproc_components[k]
            component_params = []
            for val in v.values():
                component_params.append(eval(val))
            setattr(self, k, component_class(*component_params))

    def get_feature_mat(self, additional_name=""):
        # either return feature matrix FILE or return None
        rn = self._app_entry.reranker_name
        esf_path = os.path.join('pre_processing', 'resources', 'escache', f'{rn}{additional_name}.csv')
        if os.path.exists(esf_path):
            return esf_path
        return None

    def save_feature_mat(self, fm):
        rn = self._app_entry.reranker_name
        esf_path = os.path.join('pre_processing', 'resources', 'escache', f'{rn}.csv')
        fm.to_csv(esf_path, index=False)

    def dump_svm(self, X, y, qids, docids, dense=False, zero_indexed=True, train=True):
        rn = self._app_entry.reranker_name

        if train:
            _train = 'train'
        else:
            _train = 'test'

        if zero_indexed:
            _zi = 'zero'
        else:
            _zi = 'one'

        filename = f"{rn}_{_train}_{_zi}"

        if dense:
            svm_path = os.path.join('pre_processing', 'resources', 'svmcache', f'{filename}.densesvm')

            # we manually create the dense svm file because we don't want to drop zero-values
            if os.path.exists(svm_path):
                os.remove(svm_path)
            with open(svm_path, 'a') as f:
                for i, (rel, qid, docid) in enumerate(zip(y, qids, docids)):
                    feats = X.iloc[i].to_list()
                    if zero_indexed:
                        feats = [f"{fnum}:{fval}" for fnum, fval in enumerate(feats)]
                    else:
                        feats = [f"{fnum + 1}:{fval}" for fnum, fval in enumerate(feats)]
                    f.write(' '.join([f"{rel}", f"qid:{qid}"] + feats + [f"# docid = {docid}\n"]))


        elif not dense:
            svm_path = os.path.join('pre_processing', 'resources', 'svmcache', f'{filename}.sparsesvm')
            dump_svmlight_file(X, y, svm_path, query_id=qids, zero_based=zero_indexed)
            if not docids is None:
                with open(svm_path, 'r') as f:
                    file_lines = [line.strip() for line in f.readlines()]
                    appends = [f"# docid = {docid}\n" for docid in docids]
                    zl = zip(file_lines, appends)
                    outlines = [' '.join(z) for z in zl]

                with open(svm_path, 'w') as f:
                    f.writelines(outlines)
        else:
            raise ValueError("Invalid dense/sparse option: ", dense)

    def svm_to_csv(self, svmfile, outfile):

        reader = csv.reader(
            open('pre_processing/resources/expanded_feature_numbers.csv', 'r'))
        feat_dict = {}
        for k, v in reader:
            feat_dict[k] = v

        with open(svmfile, 'r') as f:

            samples = []

            for line in f.readlines():  # note: readlines has the \n at the end
                line = line.strip()
                featvals = line.split(" ")
                rel = featvals[0]
                qid = featvals[1].split(':')[1]
                docid = featvals[-1]

                sample_dict = {}

                sample_dict['qid'] = qid
                sample_dict['doc_id'] = docid
                for featval in featvals[2:-3]:
                    k, v = featval.split(':')
                    sample_dict[feat_dict[str(int(k) + 1)]] = v

                samples.append(sample_dict)
        outdf = pd.DataFrame(samples)

        outdf.to_csv(outfile, index=False)

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
