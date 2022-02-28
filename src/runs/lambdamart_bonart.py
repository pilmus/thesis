import os

from tqdm import tqdm


# import src.evaluation.validate_run as validate
from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMart

train_dir = 'training/2019'
eval_dir = 'evaluation/2019'
for i in range(0, 1):
    for rs in [0,10,20,30,40,50,60,70,80,90,100]:
        for strat in [None, 'avg', 'dropzero']:
            for ts in [
                # 'training-sequence-1.tsv',
                'training-sequence-full.tsv',
                ]:

                print(i, rs, strat, ts)
                # OUT = os.path.join(eval_dir,f"fairRuns/submission_lambdamart-test.json")
                OUT = os.path.join(eval_dir,f"fairRuns/submission_lambdamart-{i}-{ts}-strat-{strat}-seed-{rs}.json")
                QUERIES_EVAL = os.path.join(eval_dir,"fair-TREC-evaluation-sample.json")
                SEQUENCE_EVAL = os.path.join(eval_dir,"fair-TREC-evaluation-sequences.csv")

                QUERIES_TRAIN = os.path.join(train_dir,"fair-TREC-training-sample-cleaned.json")
                SEQUENCE_TRAIN = os.path.join(train_dir, ts)

                corpus = Corpus('semanticscholar2019og')
                ft = FeatureEngineer(corpus, fquery='config/featurequery_bonart.json',
                                     fconfig='config/features_bonart.json')

                input_train = InputOutputHandler(corpus,
                                                 fsequence=SEQUENCE_TRAIN,
                                                 fquery=QUERIES_TRAIN)

                input_test = InputOutputHandler(corpus,
                                                fsequence=SEQUENCE_EVAL,
                                                fquery=QUERIES_EVAL)

                lambdamart = LambdaMart(ft, random_state=rs)
                lambdamart.train(input_train, random_state=rs,missing_value_strategy=strat)
                lambdamart.predict(input_test,missing_value_strategy=strat)

                input_test.write_submission(lambdamart, outfile=OUT)

                # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
                # validate.main(args)
