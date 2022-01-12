import src.reranker.deltr_og as deltr
from src.interface.corpus import Corpus
from src.interface.features import FeatureEngineer
from src.interface.iohandler import InputOutputHandler




QUERIES_TRAIN = "../resources/2019/training/fair-TREC-training-sample-cleaned.json"
SEQUENCE_TRAIN = "../resources/2019/training/training-sequence-handmade.tsv"

corpus = Corpus()
ft = FeatureEngineer(corpus, fquery="./config/featurequery.json", fconfig='./config/features.json')

input_train = InputOutputHandler(corpus,
                                 fsequence=SEQUENCE_TRAIN,
                                 fquery=QUERIES_TRAIN,
                                 fgroup='../resources/2019/fair-TREC-sample-author-groups.csv')


deltr = deltr.DeltrWrapper(ft)
weights = deltr.train(input_train)

