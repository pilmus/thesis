import pickle
import sys

from evaluation.twenty_twenty.merged_annotations_to_groups import group_mapping
from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.reranker.deltr_ferraro import DeltrFerraro

from src.evaluation.validate_run import validate

CORPUS = Corpus('localhost', '9200', 'semanticscholar2020')

FEATURE_ENGINEER = FeatureEngineer(CORPUS, fquery="resources/elasticsearch-ltr-config/featurequery_deltr.json",
                                   fconfig='resources/elasticsearch-ltr-config/features_deltr.json')

QUERIES_TRAIN = "resources/training/2020/DELTR-training-sample.json"
SEQUENCE_TRAIN = f"resources/training/2020/DELTR-sequence.tsv"

QUERIES_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json"
SEQUENCE_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.csv"

input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN)

input_eval = InputOutputHandler(CORPUS,
                                fsequence=SEQUENCE_EVAL,
                                fquery=QUERIES_EVAL)

training_group_file = 'resources/training/2020/doc-annotations-hclass-groups.csv'
training_feature = 'group'  # grouping has been prepared beforehand

eval_group_file = 'resources/evaluation/2020/merged-annotations-groups-mixed_group.csv'
eval_feature = 'group'

training = False
relweight = 0.5

if training:
    deltr = DeltrFerraro(FEATURE_ENGINEER, training_group_file, relevance_weight=relweight,
                         standardize=True)

    print("Training model...")
    deltr.train(input_train)
    deltr.save()


# use this if you want to continue with previously trained models
else:
    print("Loading trained models...")
    deltr_zero_pickle_path = 'resources/models/2020/deltr_gamma_0_std_True.pickle'
    deltr_one_pickle_path = 'resources/models/2020/deltr_gamma_1_std_True.pickle'
    deltr = DeltrFerraro(FEATURE_ENGINEER, eval_group_file, relevance_weight=relweight,
                         standardize=True)  # other option: instead of making whole new deltrobject just update the feature and mapping
    deltr.load(deltr_zero_pickle_path, deltr_one_pickle_path)

if training:
    deltr.grouping = eval_group_file

print("Predicting...")
deltr.predict(input_eval)

print("Writing submission...")
out = f"resources/evaluation/2020/rawruns/deltr_gammas_std_True-full-iter-5-relweight-{relweight}.json"
input_eval.write_submission(deltr, outfile=out)

print(f"Validating {out}...")
validate(QUERIES_EVAL, SEQUENCE_EVAL, out)
