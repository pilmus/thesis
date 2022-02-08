import argparse
import os.path

from src.bonart.interface.corpus import Corpus
from src.bonart.interface.features import FeatureEngineer
from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.reranker.deltr_ferraro import DeltrFerraro

from src.evaluation.validate_run import validate


def main():
    parser = argparse.ArgumentParser(description='train/evaluate with Deltr ranker')

    parser.add_argument('-c', '--corpus', dest='corpus', default='semanticscholar2020')

    parser.add_argument('--feature-query', dest='fq',
                        default="resources/elasticsearch-ltr-config/featurequery_ferraro.json")
    parser.add_argument('--feature-config', dest='fc', default='resources/elasticsearch-ltr-config/features_ferraro.json')

    parser.add_argument('--queries-train', default="resources/training/2020/DELTR-training-sample.json")
    parser.add_argument('--sequence-train', default=f"resources/training/2020/DELTR-sequence.tsv")
    parser.add_argument('--training-group-file')

    parser.add_argument('--queries-eval', default="resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json")
    parser.add_argument('--sequence-eval', default="resources/evaluation/2020/TREC-Fair-Ranking-eval-seq.tsv")
    parser.add_argument('--eval-group-file',
                        default='resources/evaluation/2020/groupings/merged-annotations-groups-mixed_group.csv')

    parser.add_argument('--deltr-zero')
    parser.add_argument('--deltr-one')

    parser.add_argument('-t', '--training', dest='training', default=False, action='store_true',
                        help='train and save a model before evaluating')
    parser.add_argument('-r', '--relevance-weight', dest='alpha')

    args = parser.parse_args()

    corp = args.corpus

    fquery = args.fq
    fconfig = args.fc

    queries_train = args.queries_train
    sequence_train = args.sequence_train
    queries_eval = args.queries_eval

    sequence_eval = args.sequence_eval
    training_group_file = args.training_group_file
    eval_group_file = args.eval_group_file

    training = args.training
    alpha = float(args.alpha)

    corpus = Corpus('localhost', '9200', corp)

    feature_engineer = FeatureEngineer(corpus, fquery=fquery,
                                       fconfig=fconfig)

    input_train = InputOutputHandler(corpus,
                                     fsequence=sequence_train,
                                     fquery=queries_train)

    input_eval = InputOutputHandler(corpus,
                                    fsequence=sequence_eval,
                                    fquery=queries_eval)

    train_group_name = os.path.basename(training_group_file).replace('doc-annotations-hclass-groups-', '').replace(
        '.csv', '')
    eval_group_name = os.path.basename(eval_group_file).replace('merged-annotations-groups-', '').replace('.csv', '')

    if training:
        deltr = DeltrFerraro(feature_engineer, training_group_file, train_group_name, alpha=alpha,
                             standardize=True)

        print("Training model...")
        deltr.train(input_train)
        deltr.save()
        deltr.grouping = eval_group_file

    # use this if you want to continue with previously trained models
    else:
        # todo; incorporate in cli
        print("Loading trained models...")
        deltr_zero_pickle_path = 'resources/models/2020/deltr_gamma_0_alpha_0.0_corp_2020subset.pickle'
        deltr_one_pickle_path = 'resources/models/2020/deltr_gamma_1_alpha_0.0_corp_2020subset.pickle'
        deltr_zero_pickle_path = args.deltr_zero
        deltr_one_pickle_path = args.deltr_one

        deltr = DeltrFerraro(feature_engineer, eval_group_file, eval_group_name, alpha=alpha,
                             standardize=True)
        deltr.load(deltr_zero_pickle_path, deltr_one_pickle_path)

    print("Predicting...")
    deltr.predict(input_eval)

    print("Writing submission...")

    out = f"resources/evaluation/2020/rawruns/deltr_gammas-alpha-{alpha}-corpus-{corp}-tgrouping-{train_group_name}-egrouping-{eval_group_name}.json"
    input_eval.write_submission(deltr, outfile=out)

    print(f"Validating {out}...")
    validate(queries_eval, sequence_eval, out)


if __name__ == '__main__':
    main()
