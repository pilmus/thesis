from src.interface.corpus import Corpus
from src.interface.features import FeatureEngineer
from src.interface.iohandler import InputOutputHandler
from src.reranker.deltr import DeltrWrapper


# protected feature name and which values of the feature are protected and which aren't
CORPUS = Corpus()
FEATURE_ENGINEER = FeatureEngineer(CORPUS, fquery="./config/featurequery_deltr.json",
                                   fconfig='./config/features_deltr.json')

PROT_MAPPING = {'feature_name': 'DocHLevel', 'value_mapping':{'H': 0, 'Mixed': 1, 'L': 1}}

SEQUENCE_TRAIN = "../resources/2019/training/training-sequence-handmade.tsv"  # todo: make sure training sequence
QUERIES_TRAIN = "../resources/2019/training/fair-TREC-training-sample-cleaned.json"

SEQUENCE_EVAL = "../resources/2019/training/training-sequence-handmade.tsv"
QUERIES_EVAL = "../resources/2019/training/fair-TREC-training-sample-cleaned.json"

input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN,
                                 fgroup='../resources/2019/fair-TREC-sample-author-groups.csv')

input_eval = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_EVAL,
                                 fquery=QUERIES_EVAL,
                                 fgroup='../resources/2019/fair-TREC-sample-author-groups.csv')

# hyperparams
gamma = 0
standardize = True

deltr = DeltrWrapper(FEATURE_ENGINEER, PROT_MAPPING, gamma, standardize=standardize)
deltr.train(input_train, save=True)
deltr.predict(input_eval)

outfile = f"./runs/deltr_gamma_{gamma}_prot_{PROT_MAPPING['protected_feature_name']}.json"
input_eval.write_submission(deltr, outfile=outfile)



