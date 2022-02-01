import pickle
import sys

from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.reranker.deltr_ferraro import DeltrFerraro

from src.evaluation.validate_run import validate

CORPUS = Corpus('localhost', '9200', 'semanticscholar2020subset')

print(CORPUS.count_docs())


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

training = True


if training:
    deltr = DeltrFerraro(FEATURE_ENGINEER, training_feature, training_group_mapping, training_group_file,
                         standardize=True)


    print("Training model...")
    deltr.train(input_train)
    deltr.save()

    deltr_non = DeltrFerraro(FEATURE_ENGINEER, training_feature, training_group_mapping, training_group_file,
                         standardize=False)
    deltr_non.train(input_train)
    deltr_non.save()

# use this if you want to continue with previously trained models
else:
    print("Loading trained models...")
    deltr_zero_pickle_path = 'resources/models/2020/deltr_gamma_0_prot_h_class_std_True.pickle'
    deltr_one_pickle_path = 'resources/models/2020/deltr_gamma_1_prot_h_class_std_True.pickle'
    deltr = DeltrFerraro(FEATURE_ENGINEER, training_feature, training_group_mapping,
                         training_group_file, standardize=True)  # other option: instead of making whole new deltrobject just update the feature and mapping
    deltr_non = DeltrFerraro(FEATURE_ENGINEER, training_feature, training_group_mapping,
                         training_group_file, standardize=False)

    deltr.load(deltr_zero_pickle_path, deltr_one_pickle_path)
    deltr_non.load('resources/models/2020/deltr_gamma_0_prot_h_class_std_False.pickle', 'resources/models/2020/deltr_gamma_1_prot_h_class_std_False.pickle')

    deltr.predict(input_train)
    deltr_non.predict(input_train)


if training:
    deltr._protected_feature = eval_feature
    deltr.protected_feature_mapping = eval_group_mapping
    deltr.grouping = eval_group_file

    deltr_non._protected_feature = eval_feature
    deltr_non.protected_feature_mapping = eval_group_mapping
    deltr_non.grouping = eval_group_file

print("Predicting...")
deltr.predict(input_eval)
deltr_non.predict(input_eval)

print("Writing submission...")
out = f"resources/evaluation/2020/rawruns/deltr_gammas_std_True_subset_training_as_eval.json"
out_non = f"resources/evaluation/2020/rawruns/deltr_gammas_std_False_subset_training_as_eval.json"
input_eval.write_submission(deltr, outfile=out)
input_eval.write_submission(deltr_non, outfile=out_non)

print(f"Validating {out}...")
validate(QUERIES_EVAL, SEQUENCE_EVAL, out)

print(f"Validating {out_non}...")
validate(QUERIES_EVAL, SEQUENCE_EVAL, out_non)