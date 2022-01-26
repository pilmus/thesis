import pickle

from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.reranker.deltr import DeltrWrapper

from src.evaluation.validate_run import validate

CORPUS = Corpus('localhost', '9200', 'semanticscholar2020')

FEATURE_ENGINEER = FeatureEngineer(CORPUS, fquery="resources/elasticsearch-ltr-config/featurequery_deltr.json",
                                   fconfig='resources/elasticsearch-ltr-config/features_deltr.json')

PROT_MAPPING = {'feature_name': 'DocHLevel', 'value_mapping': {'H': 0, 'Mixed': 1, 'L': 1}}





num_samples = 1000


QUERIES_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json"
SEQUENCE_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.csv"

QUERIES_TRAIN = "resources/training/2020/TREC-Fair-Ranking-training-sample.json"
SEQUENCE_TRAIN = f"resources/training/2020/training-sequence-{num_samples}.tsv"

# SEQUENCE_EVAL = SEQUENCE_TRAIN
# QUERIES_EVAL = QUERIES_TRAIN





input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN)

input_eval = InputOutputHandler(CORPUS,
                                fsequence=SEQUENCE_EVAL,
                                fquery=QUERIES_EVAL)

group_file = 'resources/training/2020/doc-annotations.csv'

# hyperparams
gamma = 1
standardize = True

deltr = DeltrWrapper(FEATURE_ENGINEER, PROT_MAPPING, gamma, group_file, standardize=standardize)
print("Training model...")
deltr.train(input_train)

print(f"Saving model...")
with open(f"resources/models/2020/deltr_gamma_{gamma}_prot_{PROT_MAPPING['feature_name']}_{num_samples}.pickl",'wb') as fp:
    pickle.dump(deltr.dtr, fp)


print("Predicting...")
deltr.predict(input_eval)

print("Writing submission...")

OUT = f"resources/evaluation/2020/rawruns/deltr_gamma_{gamma}_prot_{PROT_MAPPING['feature_name']}_{num_samples}.json"
input_eval.write_submission(deltr, outfile=OUT)

print(f"Validating {OUT}...")
validate(QUERIES_EVAL, SEQUENCE_EVAL, OUT)
