import os

from features.features import DeltrFeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler


def test_get_feature_matrix_merge_with_doc_annotations():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    q = os.path.join(root, 'src/features/full-sample-2020.json')
    seq = os.path.join(root, 'src/features/full-sample-2020-seq.csv')

    corpus = Corpus('semanticscholar2020og')
    esf = os.path.join(root, 'src/features/es-features-ferraro-sample-2020.csv')

    ft = DeltrFeatureEngineer(corpus, fquery=os.path.join(root, 'config', 'featurequery_ferraro_lmr.json'),
                              fconfig=os.path.join(root, 'config', 'features_ferraro_lmr.json'),
                              doc_annotations=os.path.join(root, 'src/features/doc-annotations.csv'),
                              es_feature_mat=esf)

    ioh = InputOutputHandler(corpus, seq, q)

    merged = ft.get_feature_mat(ioh)

    assert merged.columns.to_list() == ["qid", "title_score", "abstract_score", "entities_score", "venue_score",
                                        "journal_score",
                                        "authors_score", "inCitations", "outCitations", "doc_id", "qlength", "Advanced",
                                        "Developing", "DocLevel", "H", "L", "DocHLevel"]
    assert len(merged) == 3983
