import os

from tqdm import tqdm

# import src.evaluation.validate_run as validate
from interface.corpus import Corpus
from interface.features import FeatureEngineer
from interface.iohandler import InputOutputHandler
from reranker.lambdamart import LambdaMart, LambdaMartRandomization

train_dir = 'training/2020'
eval_dir = 'evaluation/2020'
idxname = 'semanticscholar2020og'
saved_features = 'src/interface/es-features-ferraro-sample-2020.csv'

for rs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for strat in [None]:
        for sort_reverse in [False, True]:
            # for sort_reverse in [False]:
            for sf in [saved_features]:
                for ts in [
                    'training-sequence-full.tsv',
                ]:
                    print(rs, strat, ts)
                    OUT = os.path.join(eval_dir, 'rawruns',
                                       f"submission_lambdamart_r-{ts}-rev-{sort_reverse}-seed-{rs}.json")
                    QUERIES_EVAL = os.path.join(eval_dir, "TREC-Fair-Ranking-eval-sample.json")
                    SEQUENCE_EVAL = os.path.join(eval_dir, "TREC-Fair-Ranking-eval-seq.tsv")

                    QUERIES_TRAIN = os.path.join(train_dir, "TREC-Fair-Ranking-training-sample.json")
                    SEQUENCE_TRAIN = os.path.join(train_dir, ts)

                    corpus = Corpus(idxname)
                    ft = FeatureEngineer(corpus, fquery='config/featurequery_ferraro_lmr.json',
                                         fconfig='config/features_ferraro_lmr.json', feature_mat=sf)

                    input_train = InputOutputHandler(corpus,
                                                     fsequence=SEQUENCE_TRAIN,
                                                     fquery=QUERIES_TRAIN)

                    input_test = InputOutputHandler(corpus,
                                                    fsequence=SEQUENCE_EVAL,
                                                    fquery=QUERIES_EVAL)

                    lambdamart = LambdaMartRandomization(ft, random_state=rs)
                    lambdamart.train(input_train)
                    lambdamart.predict(input_test)

                    input_test.write_submission(lambdamart, outfile=OUT)

                    # args = validate.Args(queries=QUERIES_EVAL, query_sequence_file = SEQUENCE_EVAL, run_file=OUT)
                    # validate.main(args)
