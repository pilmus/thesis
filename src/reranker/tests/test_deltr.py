import os

import pytest

from features.features import DeltrFeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler
from reranker.tempdeltr import DeltrWrapper


def test_prepare_data_correctly_adds_h_class():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    qtrain = os.path.join(root, 'training/2020/TREC-Fair-Ranking-training-sample.json')
    seqtrain = os.path.join(root, 'training/2020/training-sequence-full.tsv')

    qeval = os.path.join(root, 'evaluation/2020/TREC-Fair-Ranking-eval-sample.json')
    seqeval = os.path.join(root, 'evaluation/2020/TREC-Fair-Ranking-eval-seq.tsv')

    corpus = Corpus('semanticscholar2020og')
    esf = os.path.join(root, 'src/features/es-features-ferraro-sample-2020.csv')

    ft = DeltrFeatureEngineer(corpus, fquery=os.path.join(root, 'config', 'featurequery_ferraro_lmr.json'),
                              fconfig=os.path.join(root, 'config', 'features_ferraro_lmr.json'),
                              doc_annotations=os.path.join(root, 'src/features/doc-annotations.csv'),
                              es_feature_mat=esf)

    ioh_train = InputOutputHandler(corpus, seqtrain, qtrain)
    ioh_eval = InputOutputHandler(corpus, seqeval, qeval)

    deltr = DeltrWrapper(ft, 'DocHLevel', {'L': 0, 'Mixed': 0, 'H': 1})
    feats_train = deltr._prepare_data(ioh_train)
    feats_eval = deltr._prepare_data(ioh_eval, has_judgment=False)
    assert feats_train.columns.to_list() == ["q_num", "doc_id", "protected",
                                             "title_score", "abstract_score", "entities_score",
                                             "venue_score", "journal_score", "authors_score", "inCitations",
                                             "outCitations"
                                             "qlength",
                                             "relevance"]
    assert feats_eval.columns.to_list() == ["q_num", "doc_id", "protected",
                                            "title_score", "abstract_score", "entities_score",
                                            "venue_score", "journal_score", "authors_score", "inCitations",
                                            "outCitations"
                                            "qlength"]

