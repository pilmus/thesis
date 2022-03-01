import numpy as np
import pandas as pd

from interface.corpus import Corpus
from interface.features import FeatureEngineer
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


if __name__ == '__main__':
    # get_es_features('semanticscholar2019og', 'config/featurequery_bonart.json', 'config/features_bonart.json',
    #                 'src/interface/full-sample-cleaned-2019-seq.csv', 'src/interface/full-sample-cleaned-2019.jsonl',
    #                 'src/interface/es-features-bonart-sample-cleaned-2019.csv')
    get_es_features('semanticscholar2020og', 'config/featurequery_ferraro.json', 'config/features_ferraro.json',
                    'src/interface/full-sample-2020-seq.csv', 'src/interface/full-sample-2020.json',
                    'src/interface/es-features-ferraro-sample-2020.csv')
