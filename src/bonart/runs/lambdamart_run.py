import pickle

import src.bonart.reranker.lambdamart as model
from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.evaluation.twenty_nineteen.validate_run import validate

num_samples = 125000

OUT = f"resources/evaluation/2019/fairRuns/submission_lambdamart_{num_samples}.json"
QUERIES_EVAL = "resources/evaluation/2019/TREC-Competition-eval-sample-with-rel.json"
SEQUENCE_EVAL = "resources/evaluation/2019/TREC-Competition-eval-seq-5-25000.csv"

QUERIES_TRAIN = "resources/training/2019/fair-TREC-training-sample-cleaned.json"
SEQUENCE_TRAIN = f"resources/training/2019/training-sequence-{num_samples}.tsv"

CORPUS = Corpus('localhost','9200','semanticscholar')

ft = FeatureEngineer(CORPUS, fquery='resources/elasticsearch-ltr-config/featurequery.json',
                     fconfig='resources/elasticsearch-ltr-config/features.json')

input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN)

input_test = InputOutputHandler(CORPUS,
                                fsequence=SEQUENCE_EVAL,
                                fquery=QUERIES_EVAL)

lambdamart = model.LambdaMart(ft)
print("Training model...")
lambdamart.train(input_train)
print(f"Saving model...")
with open(f'resources/models/2019/lambdamart-{num_samples}.pickle','wb') as fp:
    pickle.dump(lambdamart.lambdamart, fp)
print("Predicting ranking...")
lambdamart.predict(input_test)

print("Writing submission...")
input_test.write_submission(lambdamart, outfile=OUT)
#
# # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
# # validate.main(args)


print(f"Validating {OUT}...")
validate(QUERIES_EVAL, SEQUENCE_EVAL, OUT)