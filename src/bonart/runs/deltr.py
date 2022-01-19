from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.reranker.deltr import DeltrWrapper

# from src.evaluate.twenty_twenty.validate_run_rerank import validate

CORPUS = Corpus()

FEATURE_ENGINEER = FeatureEngineer(CORPUS, fquery="resources/elasticsearch-ltr-config/featurequery_deltr.json",
                                   fconfig='resources/elasticsearch-ltr-config/features_deltr.json')

PROT_MAPPING = {'feature_name': 'DocHLevel', 'value_mapping': {'H': 0, 'Mixed': 1, 'L': 1}}

SEQUENCE_TRAIN = "resources/training/2020/training-sequence-10.tsv"  # todo: make sure training sequence
QUERIES_TRAIN = "resources/training/2020/TREC-Fair-Ranking-training-sample.json"

SEQUENCE_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.csv"
QUERIES_EVAL = "resources/evaluation/2020/TREC-Fair-Ranking-eval-sample-no-rel.json"




input_train = InputOutputHandler(CORPUS,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN)

input_eval = InputOutputHandler(CORPUS,
                                fsequence=SEQUENCE_EVAL,
                                fquery=QUERIES_EVAL)

# hyperparams
gamma = 0
standardize = True

deltr = DeltrWrapper(FEATURE_ENGINEER, PROT_MAPPING, gamma, standardize=standardize)
deltr.train(input_train, save=True)
deltr.predict(input_eval)

outfile = f"./runs/deltr_gamma_{gamma}_prot_{PROT_MAPPING['feature_name']}.json"
input_eval.write_submission(deltr, outfile=outfile)

# validate(QUERIES_EVAL, SEQUENCE_EVAL, outfile)
