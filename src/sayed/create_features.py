from bonart.interface.corpus import Corpus
from bonart.interface.features import ESFeatureEngineer
from bonart.interface.iohandler import Queries


def extract_features(fq, fc, corpus, sample):
    fe = ESFeatureEngineer(Corpus('localhost', '9200', corpus), fq, fc)
    q = Queries(sample)
    features = fe.get_feature_mat_from_queries(q)
    return features






fe = ESFeatureEngineer(Corpus('localhost', '9200', 'semanticscholar2019'), "resources/elasticsearch-ltr-config/featurequery_bonart.json", "resources/elasticsearch-ltr-config/features_bonart.json")
Queries("resources/training/2019/fair-TREC-training-sample.json")