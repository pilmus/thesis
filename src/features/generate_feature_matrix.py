import numpy as np
import pandas as pd

from features.features import FeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler


def concatenate_train_eval_queries(train_file, eval_file, outfile):
    t = pd.read_json(train_file, lines=True)
    e = pd.read_json(eval_file, lines=True)
    e = e[t.columns]
    c = pd.concat([t, e]).reset_index(drop=True)
    c.to_json(outfile, lines=True, orient='records')
    return c


def queries_to_sequence(queries, outfile):
    q = pd.read_json(queries, lines=True)
    s = q[['qid']]
    s['q_num'] = np.arange(s.shape[0])
    s['sid'] = 0
    s = s[['sid', 'q_num', 'qid']]
    s.to_csv(outfile, index=False, header=False)
    return s


def get_es_features(corp, fquery, fconfig, seq, queries, outfile):
    corpus = Corpus(corp)
    ft = FeatureEngineer(corpus, fquery=fquery,
                         fconfig=fconfig)
    ioh = InputOutputHandler(corpus,
                             fsequence=seq,
                             fquery=queries)
    feat = ft.get_feature_mat(ioh)
    feat.to_csv(outfile, index=False)
    return feat


def get_doc_annotations(annotations):
    annotations = pd.read_csv(annotations)
    return annotations


def merge_es_features_with_doc_annotations(es_features,doc_annotations,outfile):
    esdf = pd.read_csv(es_features)
    dadf = get_doc_annotations(doc_annotations)
    merged = pd.merge(esdf,dadf, left_on='doc_id',right_on='id',how='left')
    merged.to_csv(outfile,index=False)
    return merged


if __name__ == '__main__':
    merge_es_features_with_doc_annotations('es-features-ferraro-sample-2020.csv', 'doc-annotations.csv','merged-features-ferraro-2020.csv')
