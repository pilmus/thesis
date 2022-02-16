import argparse
import os

from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', dest='v', help='model version')
    parser.add_argument('-s', '--sequence', dest='s')
    parser.add_argument('-q', '--queries', dest='q')
    parser.add_argument('-o', '--outfile', dest='o')

    args = parser.parse_args()
    v = args.v
    s = args.s
    q = args.q
    o = args.o

    if v == 'bonart_lm':
        ft = FeatureEngineer(Corpus('localhost', '9200', 'semanticscholar2019'),
                             fquery="resources/elasticsearch-ltr-config/featurequery_bonart.json",
                             fconfig="resources/elasticsearch-ltr-config/features_bonart.json")
        input = InputOutputHandler(Corpus('localhost', '9200', 'semanticscholar2019'),
                                   fsequence=s,
                                   fquery=q)
    elif v == 'ferraro_lm':
        ft = FeatureEngineer(Corpus('localhost', '9200', 'semanticscholar2020'),
                             fquery="resources/elasticsearch-ltr-config/featurequery_ferraro.json",
                             fconfig="resources/elasticsearch-ltr-config/features_ferraro.json")
        input = InputOutputHandler(Corpus('localhost', '9200', 'semanticscholar2020'),
                                   fsequence=s,
                                   fquery=q)
    elif v == 'ferraro_deltr':
        ft = FeatureEngineer(Corpus('localhost', '9200', 'semanticscholar2020'),
                             fquery="resources/elasticsearch-ltr-config/featurequery_ferraro.json",
                             fconfig="resources/elasticsearch-ltr-config/features_ferraro.json")
        input = InputOutputHandler(Corpus('localhost', '9200', 'semanticscholar2020'),
                                   fsequence=s,
                                   fquery=q)
    else:
        raise ValueError(f"Invalid version: {v}.")

    features = ft.get_feature_mat(input)
    out = os.path.join('resources/features/', o)
    features.to_csv(out, index=False)


if __name__ == '__main__':
    main()
