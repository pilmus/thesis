import os

from tqdm import tqdm

# import src.evaluation.validate_run as validate
from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMartYear

train_dir = 'training/2019'
eval_dir = 'evaluation/2019'
for i in range(0, 1):
    # for rs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for rs in [0]:
        # for strat in [None, 'avg', 'dropzero']:
        for strat in [None]: # strat applies to missing year imputation strategy
            for fm in ['src/interface/es-features-bonart-sample-cleaned-2019.csv']:
                for ts in [
                    'training-sequence-full.tsv',
                ]:
                    print(i, rs, strat, ts)
                    OUT = os.path.join(eval_dir, f"fairRuns/submission_lambdamart-{i}-{ts}-strat-{strat}-seed-{rs}blorple.json")
                    QUERIES_EVAL = os.path.join(eval_dir, "fair-TREC-evaluation-sample.json")
                    SEQUENCE_EVAL = os.path.join(eval_dir, "fair-TREC-evaluation-sequences.csv")

                    QUERIES_TRAIN = os.path.join(train_dir, "fair-TREC-training-sample-cleaned.json")
                    SEQUENCE_TRAIN = os.path.join(train_dir, ts)

                    corpus = Corpus('semanticscholar2019og')
                    ft = FeatureEngineer(corpus, fquery='config/featurequery_bonart.json',
                                         fconfig='config/features_bonart.json',feature_mat=fm)

                    input_train = InputOutputHandler(corpus,
                                                     fsequence=SEQUENCE_TRAIN,
                                                     fquery=QUERIES_TRAIN)

                    input_test = InputOutputHandler(corpus,
                                                    fsequence=SEQUENCE_EVAL,
                                                    fquery=QUERIES_EVAL)

                    lambdamart = LambdaMartYear(ft, random_state=rs, missing_value_strategy=strat)
                    lambdamart.train(input_train)
                    lambdamart.predict(input_test)

                    input_test.write_submission(lambdamart, outfile=OUT)

                    # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
                    # validate.main(args)
