import os
import sys

from tqdm import tqdm

# import src.evaluation.validate_run as validate
from features.features import FeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMart, LambdaMartRandomization, LambdaMartYear

train_dir = 'training/2020'
eval_dir = 'evaluation/2020'
idxname = 'semanticscholar2020og'
saved_features = 'src/features/es-features-ferraro-sample-2020.csv'
eval_seq = "TREC-Fair-Ranking-eval-seq-test.tsv"

for rs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for strat in [None]:
        for sort_reverse in [True, False]:
            for sf in [saved_features]:
                for ts in [
                    'training-sequence-full.tsv',
                ]:
                    print(rs, strat, ts)
                    OUT = os.path.join(eval_dir, 'rawruns',
                                       f"submission_lambdamart_r-{ts}-{eval_seq}-rev-{sort_reverse}-seed-{rs}.json")
                    QUERIES_EVAL = os.path.join(eval_dir, "TREC-Fair-Ranking-eval-sample.json")
                    SEQUENCE_EVAL = os.path.join(eval_dir, eval_seq)

                    QUERIES_TRAIN = os.path.join(train_dir, "TREC-Fair-Ranking-training-sample.json")
                    SEQUENCE_TRAIN = os.path.join(train_dir, ts)

                    corpus = Corpus(idxname)
                    ft = FeatureEngineer(corpus, fquery='config/featurequery_ferraro_lmr.json',
                                         fconfig='config/features_ferraro_lmr.json', feature_mat=sf)

                    input_train = InputOutputHandler(
                                                     fsequence=SEQUENCE_TRAIN,
                                                     fquery=QUERIES_TRAIN)

                    input_test = InputOutputHandler(
                                                    fsequence=SEQUENCE_EVAL,
                                                    fquery=QUERIES_EVAL)

                    # lambdamart = LambdaMartRandomization(ft, random_state=rs, sort_reverse=sort_reverse)
                    lambdamart = LambdaMartYear(ft, random_state=rs, missing_value_strategy=None)
                    lambdamart.train(input_train)
                    lambdamart.predict(input_test)

                    input_test.write_submission(lambdamart, outfile=OUT)
                    sys.exit()

                    # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
                    # validate.main(args)
