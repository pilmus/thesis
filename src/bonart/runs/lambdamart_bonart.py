import os

from tqdm import tqdm


# import src.evaluation.validate_run as validate
from bonart.interface.corpus import Corpus
from bonart.interface.features import FeatureEngineer
from bonart.interface.iohandler import InputOutputHandler
from bonart.reranker.lambdamart import LambdaMart

train_dir = 'training/2019'
eval_dir = 'evaluation/2019'
for i in range(0, 1):
    for ts in [
        # 'training-sequence-100.tsv',
        #        'training-sequence-1000.tsv',
        #        'training-sequence-1000-1-index.tsv',
        #        'training-sequence-1000-2-index.tsv',
        'training-sequence-full.tsv',
        # 'training-sequence-full-1-index.tsv',
        # 'training-sequence-full-2-index.tsv',
        ]:

        print(i, ts)
        OUT = os.path.join(eval_dir,f"fairRuns/submission_lambdamart-{ts}-seed-0.json")
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

        lambdamart = LambdaMart(ft, random_state=0)
        lambdamart.train(input_train, random_state=0)
        lambdamart.predict(input_test)

        input_test.write_submission(lambdamart, outfile=OUT)

        # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
        # validate.main(args)
