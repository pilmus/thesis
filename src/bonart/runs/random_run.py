import sys

from src.evaluation.twenty_nineteen.validate_run import validate

import src.bonart.reranker.model as model
from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler

OUT = "resources/evaluation/2019/fairRuns/submission_random_125000.json"
QUERIES = "resources/evaluation/2019/TREC-Competition-eval-sample-with-rel.json"
SEQUENCE = "resources/evaluation/2019/TREC-Competition-eval-seq-5-25000.csv"


print("Initializing corpus.")
corpus = Corpus('localhost', '9200', 'semanticscholar')
print("Building features.")
ft = FeatureEngineer(corpus, fquery='resources/elasticsearch-ltr-config/featurequery_lambdamart.json',
                     fconfig='resources/elasticsearch-ltr-config/features_lambdamart.json')

input = InputOutputHandler(corpus,
                           fsequence=SEQUENCE,
                           fquery=QUERIES)
print("Predicting...")
random = model.RandomRanker(ft)
random.predict(input)

print("Writing submission.")
input.write_submission(random, outfile=OUT)

print(f"Validating {OUT}...")
validate(QUERIES, SEQUENCE, OUT)
