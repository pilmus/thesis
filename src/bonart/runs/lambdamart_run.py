import argparse
import os.path

from src.bonart.reranker.lambdamart import LambdaMart
from src.bonart.reranker.lambdamart import LambdaMartFerraro
from src.evaluation.validate_run import validate
from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler


def main():
    parser = argparse.ArgumentParser(description='train/evaluate lambdamart')

    parser.add_argument('-c', '--corpus', dest='corpus', default='semanticscholar2020')

    parser.add_argument('-l', '--lambdamart-version', dest='lamb_vers')

    parser.add_argument('--feature-query', dest='fq',
                        default='resources/elasticsearch-ltr-config/featurequery_ferraro.json')
    parser.add_argument('--feature-config', dest='fc',
                        default='resources/elasticsearch-ltr-config/features_ferraro.json')

    parser.add_argument('--queries-train', default='resources/training/2020/TREC-Fair-Ranking-training-sample.json')
    parser.add_argument('--sequence-train', default='resources/training/2020/training-sequence-1000.tsv')

    parser.add_argument('--queries-eval', default='resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json')
    parser.add_argument('--sequence-eval', default="resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.tsv")

    parser.add_argument('-t', '--training', dest='training', default=False, action='store_true',
                        help='train and save a model before evaluating')

    parser.add_argument('--sort-reverse', default=False, action='store_true',
                        help='applicable for ferraro-runs. if true, sort relevances in descending order during randomization')

    parser.add_argument('--model-dir', dest='model_dir', default='resources/models',
                        help='location where the models are saved')

    args = parser.parse_args()
    corp = args.corpus

    lversion = args.lamb_vers
    fquery = args.fq
    fconfig = args.fc
    model_dir = args.model_dir

    queries_train = args.queries_train
    sequence_train = args.sequence_train
    queries_eval = args.queries_eval
    sequence_eval = args.sequence_eval

    training = args.training
    sort_reverse = args.sort_reverse

    num_samples_t = os.path.splitext(os.path.basename(sequence_train))[0].split('-')[-1]
    num_samples_e = os.path.splitext(os.path.basename(sequence_eval))[0].split('-')[-1]

    corpus = Corpus('localhost', '9200', corp)

    ft = FeatureEngineer(corpus, fquery=fquery,
                         fconfig=fconfig)

    input_train = InputOutputHandler(corpus,
                                     fsequence=sequence_train,
                                     fquery=queries_train)

    input_test = InputOutputHandler(corpus,
                                    fsequence=sequence_eval,
                                    fquery=queries_eval)

    if lversion == 'ferraro':
        lambdamart = LambdaMartFerraro(ft, sort_reverse)
        outfile = f"resources/evaluation/2020/rawruns/lambdamart_{corp}_{num_samples_t}_{num_samples_e}_rev_{sort_reverse}.json"
        model_path = os.path.join(model_dir, "2020")
    elif lversion == 'bonart':
        lambdamart = LambdaMart(ft)
        outfile = f"resources/evaluation/2019/fairRuns/lambdamart_bonart_{corp}_{num_samples_t}.json"
        model_path = os.path.join(model_dir, "2019")
    else:
        raise ValueError(
            f'Invalid option given for LambdaMart version: {lversion}.\nValid options are: "ferraro", "bonart".')

    model_path = os.path.join(model_path, f"lambdamart_{lversion}_{corp}_{num_samples_t}.pickle")

    if bool(training):
        print("Training model...")
        lambdamart.train(input_train)
        lambdamart.save(model_path)
    else:
        lambdamart.load(model_path)

    print("Predicting ranking...")
    lambdamart.predict(input_test)

    print("Writing submission...")
    input_test.write_submission(lambdamart, outfile=outfile)

    print(f"Validating {outfile}...")
    validate(queries_eval, sequence_eval, outfile)


if __name__ == '__main__':
    main()
