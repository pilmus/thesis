import src.reranker.lambdamart as model
from src.interface.corpus import Corpus
from src.interface.features import FeatureEngineer
from src.interface.iohandler import InputOutputHandler
from evaluation.validate_run import validate


OUT = "./evaluation/submission_lambdamart_missing_gone.json"
QUERIES_EVAL = "./evaluation/fair-TREC-evaluation-sample.json"
SEQUENCE_EVAL = "./evaluation/fair-TREC-evaluation-sequences.csv"

QUERIES_TRAIN = "./training/fair-TREC-training-sample-cleaned.json"
SEQUENCE_TRAIN = "./training/training-sequence-full.tsv"

corpus = Corpus()
ft = FeatureEngineer(corpus)

input_train = InputOutputHandler(corpus,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN)

input_test = InputOutputHandler(corpus,
                                fsequence=SEQUENCE_EVAL,
                                fquery=QUERIES_EVAL)

lambdamart = model.LambdaMart(ft)
lambdamart.train(input_train)
lambdamart.predict(input_test)

input_test.write_submission(lambdamart, outfile=OUT)
#
# # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
# # validate.main(args)


print(f"Validating {OUT}...")
validate(QUERIES_EVAL, SEQUENCE_EVAL, OUT)
