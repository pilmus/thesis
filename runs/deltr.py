import os
import sys

from features.features import AnnotationFeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler
from reranker.deltr import Deltr

train_dir = 'training/2020'
eval_dir = 'evaluation/2020'
idxname = 'semanticscholar2020og'
saved_features = 'features/es-features-ferraro-sample-2020.csv'
eval_seq = "TREC-Fair-Ranking-eval-seq.tsv"
ts = 'training-sequence-full.tsv'
doc_annotations = 'features/doc-annotations.csv'
pr = 'DocHLevel'
pr_mapping = {'L': 1, 'Mixed': 1, 'H': 0}
gamma1 = 1
gamma2 = 0
iternum = 5  # may need to change later?
alpha = 0.5

for rs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    OUT = os.path.join(eval_dir, 'rawruns',
                       f"submission_deltr-{ts}-alpha-{alpha}-seed-{rs}-pr-{pr}-prg-{'_'.join([k for k, v in pr_mapping.items() if v == 1])}-inum-{iternum}.json")
    # OUT = 'binini.json'

    QUERIES_EVAL = os.path.join(eval_dir, "TREC-Fair-Ranking-eval-sample.json")
    SEQUENCE_EVAL = os.path.join(eval_dir, eval_seq)

    QUERIES_TRAIN = os.path.join(train_dir, "TREC-Fair-Ranking-training-sample.json")
    SEQUENCE_TRAIN = os.path.join(train_dir, ts)

    corpus = Corpus(idxname)
    ft = AnnotationFeatureEngineer(doc_annotations, es_feature_mat=saved_features)

    input_train = InputOutputHandler(fsequence=SEQUENCE_TRAIN,
                                     fquery=QUERIES_TRAIN)

    input_test = InputOutputHandler(fsequence=SEQUENCE_EVAL,
                                    fquery=QUERIES_EVAL)

    deltr = Deltr(ft, pr, pr_mapping, gamma1=gamma1, num_iter=iternum, random_state=rs, gamma2=gamma2, alpha=alpha)
    deltr.train(input_train)
    deltr.predict(input_test)

    input_test.write_submission(deltr, outfile=OUT)
    sys.exit()

    # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
    # validate.main(args)
