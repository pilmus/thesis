# import evaluation.validate_run as validate

from evaluation.validate_run import validate

import src.reranker.model as model
from src.interface.corpus import Corpus
from src.interface.features import FeatureEngineer
from src.interface.iohandler import InputOutputHandler

OUT = "./evaluation/submission_random.json"
QUERIES = "./evaluation/fair-TREC-evaluation-sample.json"
SEQUENCE = "./evaluation/fair-TREC-evaluation-sequences.csv"

print("Initializing corpus.")
corpus = Corpus()
print("Building features.")
ft = FeatureEngineer(corpus)

input = InputOutputHandler(corpus,
                           fsequence=SEQUENCE,
                           fquery=QUERIES)
print("Predicting...")
random = model.RandomRanker(ft)
random.predict(input)

print("Writing submission.")
input.write_submission(random, outfile=OUT)

print(f"Validating {OUT}...")
validate(QUERIES,SEQUENCE,OUT)
# args = validate.Args(queries=QUERIES, query_sequence_file=SEQUENCE, run_file=OUT)
# validate.main(args)
