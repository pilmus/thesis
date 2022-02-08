import argparse

from bonart.reranker.lambdamart_ferraro import LambdaMartFerraro
from evaluation.validate_run import validate
from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler


def main():
    parser = argparse.ArgumentParser(description='train/evaluate with Lambdamart and randomization')

    parser.add_argument('-c', '--corpus', dest='corpus', default='2020')

    parser.add_argument('--feature-query', dest='fq',
                        default="resources/elasticsearch-ltr-config/featurequery_ferraro.json")
    parser.add_argument('--feature-config', dest='fc',
                        default='resources/elasticsearch-ltr-config/deltr.json')

    parser.add_argument('--queries-train', default="resources/training/2020/TREC-Fair-Ranking-training-sample.json")
    parser.add_argument('--sequence-train', default=f"resources/training/2020/training-sequence-10.tsv")

    parser.add_argument('--model-path', dest='model_path', default='tempname.pickle',
                        help='location where the model is saved')

    parser.add_argument('--queries-eval', default="resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json")
    parser.add_argument('--sequence-eval', default="resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.tsv")

    parser.add_argument('-t', '--training', dest='training', default=False, action='store_true',
                        help='train and save a model before evaluating')
    #todo: trainign with argument that's also save location?

    args = parser.parse_args()

    corp = args.corpus

    fquery = args.fq
    fconfig = args.fc

    model_path = args.model_path

    queries_train = args.queries_train
    sequence_train = args.sequence_train
    queries_eval = args.queries_eval
    sequence_eval = args.sequence_eval

    training = args.training

    corpus = Corpus('localhost', '9200', f'semanticscholar{corp}')

    feature_engineer = FeatureEngineer(corpus, fquery=fquery,
                                       fconfig=fconfig)

    input_train = InputOutputHandler(corpus,
                                     fsequence=sequence_train,
                                     fquery=queries_train)

    input_eval = InputOutputHandler(corpus,
                                    fsequence=sequence_eval,
                                    fquery=queries_eval)

    if training:
        model = LambdaMartFerraro(feature_engineer)

        print("Training model...")
        model.train(input_train)
        model.save(model_path)

    # use this if you want to continue with previously trained models
    else:
        # todo; incorporate in cli
        print("Loading trained model...")

        model = LambdaMartFerraro(feature_engineer)
        model.load(model_path)

    print("Predicting...")
    model.predict(input_eval)

    print("Writing submission...")
    out = f"resources/evaluation/2020/rawruns/lambdamart_ferraro.json"
    input_eval.write_submission(model, outfile=out)

    print(f"Validating {out}...")
    validate(queries_eval, sequence_eval, out)


if __name__ == '__main__':
    main()
