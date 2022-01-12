from src.interface.corpus import Corpus
from src.interface.features import FeatureEngineer
from src.interface.iohandler import InputOutputHandler
from src.reranker.deltr import DeltrWrapper


def train():
    QUERIES_TRAIN = "../resources/2019/training/fair-TREC-training-sample-cleaned.json"
    SEQUENCE_TRAIN = "../resources/2019/training/training-sequence-handmade.tsv"  # todo: make sure training sequence
    # ids
    # in sample

    corpus = Corpus()
    ft = FeatureEngineer(corpus, fquery="./config/featurequery_deltr.json", fconfig='./config/features_deltr.json')

    input_train = InputOutputHandler(corpus,
                                     fsequence=SEQUENCE_TRAIN,
                                     fquery=QUERIES_TRAIN,
                                     fgroup='../resources/2019/fair-TREC-sample-author-groups.csv')

    deltr = DeltrWrapper(ft, "h_score", 0, standardize=False)
    weights = deltr.train(input_train)
    return deltr


def eval(deltr=None):
    OUT = "./runs/submission_deltr_gamma_0.json"
    QUERIES_EVAL = "../resources/2019/training/fair-TREC-training-sample-cleaned.json"
    SEQUENCE_EVAL = "../resources/2019/training/training-sequence-handmade.tsv"

    corpus = Corpus()
    ft = FeatureEngineer(corpus, fquery="./config/featurequery_deltr.json", fconfig='./config/features_deltr.json')

    input_eval = InputOutputHandler(corpus,fsequence=SEQUENCE_EVAL,fquery =QUERIES_EVAL,fgroup='../resources/2019/fair-TREC-sample-author-groups.csv')

    if not deltr:
        deltr = DeltrWrapper(ft, "h_score", 0, standardize=False) #todo: read this initialization from file
        deltr.load_model('./models/deltr_gamma_0.model.json')
        deltr.dtr._protected_feature = 0
    deltr.predict(input_eval)

    input_eval.write_submission(deltr, outfile=OUT)





# deltr = train()
eval()
