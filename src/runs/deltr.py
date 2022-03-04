import os
import sys

from tqdm import tqdm

# import src.evaluation.validate_run as validate
from features.features import FeatureEngineer, AnnotationFeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMart, LambdaMartRandomization
from reranker.tempdeltr import Deltr

train_dir = 'training/2020'
eval_dir = 'evaluation/2020'
idxname = 'semanticscholar2020og'
saved_features = 'src/features/es-features-ferraro-sample-2020.csv'
eval_seq = "TREC-Fair-Ranking-eval-seq-test.tsv"
doc_annotations = 'src/features/doc-annotations.csv'
pr = 'DocHLevel'
pr_mapping = {'L': 1, 'Mixed': 1, 'H': 0}
gamma = 1
iternum = 5

for rs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for strat in [None]:
        for sort_reverse in [True, False]:
            for sf in [saved_features]:
                for ts in [
                    'training-sequence-full.tsv',
                ]:
                    print(rs, strat, ts)
                    OUT = os.path.join(eval_dir, 'rawruns',
                                       f"submission_deltr-{ts}-{eval_seq}-gamma-{gamma}-seed-{rs}.json")

                    QUERIES_EVAL = os.path.join(eval_dir, "TREC-Fair-Ranking-eval-sample.json")
                    SEQUENCE_EVAL = os.path.join(eval_dir, eval_seq)

                    QUERIES_TRAIN = os.path.join(train_dir, "TREC-Fair-Ranking-training-sample.json")
                    SEQUENCE_TRAIN = os.path.join(train_dir, ts)

                    corpus = Corpus(idxname)
                    ft = AnnotationFeatureEngineer(doc_annotations, es_feature_mat=sf)

                    input_train = InputOutputHandler(fsequence=SEQUENCE_TRAIN,
                                                     fquery=QUERIES_TRAIN)

                    input_test = InputOutputHandler(fsequence=SEQUENCE_EVAL,
                                                    fquery=QUERIES_EVAL)

                    deltr = Deltr(ft, pr, pr_mapping, gamma, iternum, random_state=rs)
                    deltr.train(input_train)
                    deltr.predict(input_test)

                    input_test.write_submission(deltr, outfile=OUT)
                    sys.exit()

                    # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
                    # validate.main(args)
