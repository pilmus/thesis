import pickle
import sys

from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.reranker.deltr_ferraro import DeltrFerraro

from src.evaluation.validate_run import validate

CORPUS = Corpus('localhost', '9200', 'semanticscholar2020')

FEATURE_ENGINEER = FeatureEngineer(CORPUS, fquery="resources/elasticsearch-ltr-config/featurequery_deltr.json",
                                   fconfig='resources/elasticsearch-ltr-config/features_deltr.json')

QUERIES_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json"
SEQUENCE_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.csv"

QUERIES_TRAIN = "resources/training/2020/DELTR-training-sample.json"
SEQUENCE_TRAIN = f"resources/training/2020/DELTR-sequence.tsv"



input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN)

input_eval = InputOutputHandler(CORPUS,
                                fsequence=SEQUENCE_EVAL,
                                fquery=QUERIES_EVAL)

training_group_file = 'resources/training/2020/doc-hclass.csv'
training_group_mapping = {'H': 0, 'L': 1}  # 1 is protected group as per paper
training_feature = 'h_class'

eval_group_file = 'resources/evaluation/2020/doc-annotations-groups.csv'
eval_group_mapping = {'1|2': 1, '1': 1,
                      '2': 0}  # from doc_annotations_to_groups.py: {'Advanced' : '2', 'Developing' : '1', 'Mixed': '1|2'}
eval_feature = 'group'

print("TRAINING")

deltr = DeltrFerraro(FEATURE_ENGINEER, training_feature, training_group_mapping, training_group_file)

# print("Training model...")
deltr.train(input_train)

deltr_zero_pickle_path, deltr_one_pickle_path = deltr.save()

# use this if you want to continue with previously trained models
# print("Loading trained models...")
# deltr_zero_pickle_path = 'resources/models/2020/deltr_gamma_0_prot_h_class.pickle'
# deltr_one_pickle_path = 'resources/models/2020/deltr_gamma_1_prot_h_class.pickle'
# deltr = DeltrFerraro(FEATURE_ENGINEER, eval_feature, eval_group_mapping, eval_group_file) # other option: instead of making whole new deltrobject just update the feature and mapping
# deltr.load(deltr_zero_pickle_path, deltr_one_pickle_path)

deltr._protected_feature = eval_feature
deltr.protected_feature_mapping = eval_group_mapping
deltr.grouping = eval_group_file

print("Predicting...")
deltr.predict(input_eval)

print("Writing submission...")
OUT = f"resources/evaluation/2020/rawruns/deltr_gammas.json"
input_eval.write_submission(deltr, outfile=OUT)

print(f"Validating {OUT}...")
validate(QUERIES_EVAL, SEQUENCE_EVAL, OUT)
