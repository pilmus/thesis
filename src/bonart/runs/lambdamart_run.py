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

    parser.add_argument('-c', '--corpus', dest='corpus')

    parser.add_argument('-l', '--lambdamart-version', dest='lamb_vers')

    parser.add_argument('--queries-train', default='TREC-Fair-Ranking-training-sample.json')
    parser.add_argument('--sequence-train', default='training-sequence-1000.tsv')

    parser.add_argument('--queries-eval', default='TREC-Fair-Ranking-eval-sample.json')
    parser.add_argument('--sequence-eval', default="TREC-Fair-Ranking-eval-seq.tsv")

    parser.add_argument('--features-train')
    parser.add_argument('--features-eval')

    parser.add_argument('-m', '--load-model', dest ='load_model')


    # parser.add_argument('-t', '--training', dest='training', default=False, action='store_true',
    #                     help='train and save a model before evaluating')

    parser.add_argument('--sort-reverse', default=False, action='store_true',
                        help='applicable for ferraro-runs. if true, sort relevances in descending order during randomization')


    args = parser.parse_args()


    corp = args.corpus

    # queries_train = args.queries_train
    # sequence_train = args.sequence_train
    # queries_eval = args.queries_eval
    # sequence_eval = args.sequence_eval

    features_train = args.features_train
    features_eval = args.features_eval

    load_model = args.load_model

    lversion = args.lamb_vers


    if lversion == 'ferraro':
        train_base = 'resources/training/2020/'
        eval_base = 'resources/evaluation/2020/'

        fquery = 'resources/elasticsearch-ltr-config/featurequery_ferraro.json'
        fconfig = 'resources/elasticsearch-ltr-config/features_ferraro.json'
        model_dir = 'resources/models/2020/lm_ferraro'

        sort_reverse = args.sort_reverse
        if not corp:
            corp = 'semanticscholar2020'
    elif lversion == 'bonart':
        train_base = 'resources/training/2019/'
        eval_base = 'resources/evaluation/2019/'

        fquery = 'resources/elasticsearch-ltr-config/featurequery_bonart_og.json'
        fconfig = 'resources/elasticsearch-ltr-config/features_bonart_og.json'
        model_dir = 'resources/models/2019/lm_bonart'
        if not corp:
            corp = 'semanticscholar2019og'
    else:
        raise ValueError(f"Invalid value for '-l', '--lambdamart-version': {lversion}.")

    queries_train = os.path.join(train_base, args.queries_train)
    sequence_train = os.path.join(train_base, args.sequence_train)
    queries_eval = os.path.join(eval_base, args.queries_eval)
    sequence_eval = os.path.join(eval_base, args.sequence_eval)

    print(corp)


    num_samples_t = os.path.splitext(os.path.basename(sequence_train))[0].split('-')[-1]
    num_samples_e = os.path.splitext(os.path.basename(sequence_eval))[0].split('-')[-1]

    corpus = Corpus('localhost', '9200', corp)

    engineer = FeatureEngineer(corpus, fquery=fquery,
                         fconfig=fconfig)

    input_train = InputOutputHandler(corpus,
                                     fsequence=sequence_train,
                                     fquery=queries_train)

    input_test = InputOutputHandler(corpus,
                                    fsequence=sequence_eval,
                                    fquery=queries_eval)

    if lversion == 'ferraro':
        lambdamart = LambdaMartFerraro(engineer, sort_reverse)
        outfile = f"resources/evaluation/2020/rawruns/lmf_{corp}_{num_samples_t}_{num_samples_e}_rev_{sort_reverse}.json"
    elif lversion == 'bonart':
        lambdamart = LambdaMart(engineer)
        outfile = f"resources/evaluation/2019/fairRuns/lmb_{corp}_{num_samples_t}_{num_samples_e}.json"
    else:
        raise ValueError(
            f'Invalid option given for LambdaMart version: {lversion}.\nValid options are: "ferraro", "bonart".')

    model_path = os.path.join(model_dir, f"lambdamart_{lversion}_{corp}_{num_samples_t}.pickle")

    if not load_model:
        print("Training model...")
        lambdamart.train(input_train, features_train)
        lambdamart.save(model_path)
    else:
        lambdamart.load(load_model)

    print("Predicting ranking...")
    lambdamart.predict(input_test, features_eval)

    print("Writing submission...")
    input_test.write_submission(lambdamart, outfile=outfile)

    print(f"Validating {outfile}...")
    validate(queries_eval, sequence_eval, outfile)


if __name__ == '__main__':
    main()
