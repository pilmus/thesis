import os

from features.features import AnnotationFeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler


def test_get_feature_matrix_merge_with_doc_annotations_dochlevel():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    q = os.path.join(root, 'src/features/full-sample-2020.json')
    seq = os.path.join(root, 'src/features/full-sample-2020-seq.csv')

    esf = os.path.join(root, 'src/features/es-features-ferraro-sample-2020.csv')

    ft = AnnotationFeatureEngineer(
        doc_annotations=os.path.join(root, 'src/features/doc-annotations.csv'),
        es_feature_mat=esf)

    ioh = InputOutputHandler(seq, q)

    merged = ft.get_feature_mat(ioh, annotation_features=['DocHLevel'])

    assert merged.columns.to_list() == ["qid", "title_score", "abstract_score", "entities_score", "venue_score",
                                        "journal_score",
                                        "authors_score", "inCitations", "outCitations", "doc_id", "qlength",
                                        "DocHLevel"]
    assert len(merged) == 4210


def test_get_feature_matrix_merge_training_sample_with_doc_annotations_dochlevel():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    q = os.path.join(root, 'training/2020/TREC-Fair-Ranking-training-sample.json')
    seq = os.path.join(root, 'training/2020/training-sequence-full.tsv')

    esf = os.path.join(root, 'src/features/es-features-ferraro-sample-2020.csv')

    ft = AnnotationFeatureEngineer(
        doc_annotations=os.path.join(root, 'src/features/doc-annotations.csv'),
        es_feature_mat=esf)

    ioh = InputOutputHandler(seq, q)

    merged = ft.get_feature_mat(ioh, annotation_features=['DocHLevel'])

    assert merged.columns.to_list() == ["qid", "title_score", "abstract_score", "entities_score", "venue_score",
                                        "journal_score",
                                        "authors_score", "inCitations", "outCitations", "doc_id", "qlength",
                                        "DocHLevel"]
    assert len(merged) == 2049


def test_get_feature_matrix_merge_eval_sample_with_doc_annotations_dochlevel():
    root = '/mnt/c/Users/maaik/Documents/thesis'
    q = os.path.join(root, 'evaluation/2020/TREC-Fair-Ranking-eval-sample.json')
    seq = os.path.join(root, 'evaluation/2020/TREC-Fair-Ranking-eval-seq.tsv')

    esf = os.path.join(root, 'src/features/es-features-ferraro-sample-2020.csv')

    ft = AnnotationFeatureEngineer(
        doc_annotations=os.path.join(root, 'src/features/doc-annotations.csv'),
        es_feature_mat=esf)

    ioh = InputOutputHandler(seq, q)

    merged = ft.get_feature_mat(ioh, annotation_features=['DocHLevel'])

    assert merged.columns.to_list() == ["qid", "title_score", "abstract_score", "entities_score", "venue_score",
                                        "journal_score",
                                        "authors_score", "inCitations", "outCitations", "doc_id", "qlength",
                                        "DocHLevel"]
    assert len(merged) == 2161
