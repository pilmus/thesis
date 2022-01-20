import src.bonart.reranker.lambdamart as model
from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.evaluation.twenty_nineteen.validate_run import validate


OUT = "resources/evaluation/2019/fairRuns/submission_lambdamart_missing_gone.json"
QUERIES_EVAL = "resources/evaluation/2019/fair-TREC-evaluation-sample.json"
SEQUENCE_EVAL = "resources/evaluation/2019/fair-TREC-evaluation-sequences.csv"

QUERIES_TRAIN = "resources/training/2019/fair-TREC-training-sample-cleaned.json"
SEQUENCE_TRAIN = "resources/training/2019/training-sequence-full.tsv"

CORPUS = Corpus('localhost','9200','semanticscholar')

ft = FeatureEngineer(CORPUS, fquery='resources/elasticsearch-ltr-config/featurequery.json',
                     fconfig='resources/elasticsearch-ltr-config/features_deltr.json')

input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN)

input_test = InputOutputHandler(CORPUS,
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
