from bonart.src.interface.corpus import Corpus
from bonart.src.interface.features import FeatureEngineer
from bonart.src.interface.iohandler import InputOutputHandler
from bonart.src.reranker.deltr import DeltrWrapper

from evaluate.twenty_nineteen.validate_run import validate

CORPUS = Corpus()
FEATURE_ENGINEER = FeatureEngineer(CORPUS, fquery="./config/featurequery_deltr.json",
                                   fconfig='./config/features_deltr.json')

PROT_MAPPING = {'feature_name': 'DocHLevel', 'value_mapping': {'H': 0, 'Mixed': 1, 'L': 1}}

SEQUENCE_TRAIN = "resources/training/2020/training-sequence-10.tsv"  # todo: make sure training sequence
QUERIES_TRAIN = "resources/training/2020/TREC-Fair-Ranking-training-sample.json"

SEQUENCE_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.csv"
QUERIES_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-sample-no-rel.json"


# QUERIES_EVAL = "./evaluation/fair-TREC-evaluation-sample.json"
# SEQUENCE_EVAL = "./evaluation/fair-TREC-evaluation-sequences.csv"
#
# QUERIES_TRAIN = "./training/fair-TREC-training-sample-cleaned.json"
# SEQUENCE_TRAIN = "./training/training-sequence-full.tsv"


input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN,
                                 fgroup='../../resources/2019/fair-TREC-sample-author-groups.csv')

input_eval = InputOutputHandler(CORPUS,
                                fsequence=SEQUENCE_EVAL,
                                fquery=QUERIES_EVAL,
                                fgroup='../../resources/2019/fair-TREC-sample-author-groups.csv')

# hyperparams
gamma = 0
standardize = True

deltr = DeltrWrapper(FEATURE_ENGINEER, PROT_MAPPING, gamma, standardize=standardize)
deltr.train(input_train, save=True)
deltr.predict(input_eval)

outfile = f"./runs/deltr_gamma_{gamma}_prot_{PROT_MAPPING['feature_name']}.json"
input_eval.write_submission(deltr, outfile=outfile)

validate(QUERIES_EVAL, SEQUENCE_EVAL, outfile)
