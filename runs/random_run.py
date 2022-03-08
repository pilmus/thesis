import argparse

from src.bonart.interface.corpus import Corpus
from src.bonart.interface.iohandler import InputOutputHandler
from src.bonart.reranker.random_ranker import RandomRanker
from evaluation.validate_run import validate


def main():
    parser = argparse.ArgumentParser(description='random run')
    parser.add_argument('-c', '--corpus', dest='corpus', default='semanticscholar2019')
    parser.add_argument('-q', '--queries', dest='queries',
                        default="resources/evaluation/2019/TREC-Competition-eval-sample-with-rel.json")
    parser.add_argument('-s', '--sequence', dest='sequence',
                        default="resources/evaluation/2019/TREC-Competition-eval-seq-5-25000.csv")
    parser.add_argument('-o', '--out', dest='out', default="resources/evaluation/2019/fairRuns/submission_random_125000.json")

    args = parser.parse_args()

    corpus = args.corpus
    queries = args.queries
    sequence = args.sequence
    out = args.out

    print("Initializing corpus.")
    corpus = Corpus('localhost', '9200', corpus)

    input = InputOutputHandler(corpus,
                               fsequence=sequence,
                               fquery=queries)
    print("Predicting...")
    random = RandomRanker(None)
    random.predict(input)

    print("Writing submission.")
    input.write_submission(random, outfile=out)

    print(f"Validating {out}...")
    validate(queries, sequence, out)


if __name__ == '__main__':
    main()
