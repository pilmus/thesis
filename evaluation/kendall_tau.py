import argparse

import pandas as pd
import scipy.stats
from tqdm import tqdm

tqdm.pandas()


def compare_items_in_rankings(runfile1, runfile2):
    """Check if rankings contain same items for each qnum/qid combination."""
    print(runfile1)
    df1 = pd.read_json(runfile1, lines=True, dtype={'q_num': str, 'qid': int})
    print(runfile2)
    df2 = pd.read_json(runfile2, lines=True, dtype={'q_num': str, 'qid': int})

    df = pd.merge(df1, df2, on=['q_num', 'qid'], suffixes=['_1', '_2'])

    return len(df[df.apply(
        lambda row: not (set(row.ranking_1).issubset(row.ranking_2) & set(row.ranking_1).issubset(row.ranking_2)),
        axis=1)]) == 0


def find_ranklist_size(runfile):
    df = pd.read_json(runfile, lines=True, dtype={'q_num': str, 'qid': int})
    min = df.apply(lambda row: len(row.ranking), axis=1).min()
    max = df.apply(lambda row: len(row.ranking), axis=1).max()
    mean = df.apply(lambda row: len(row.ranking), axis=1).mean()

    return min, max, mean


def to_ranking(row):
    return list(range(0, len(row.ranking_1))), [row.ranking_1.index(item) for item in row.ranking_2]


def kendalls_taus(runfile1, runfile2):
    df1 = pd.read_json(runfile1, lines=True, dtype={'q_num': str, 'qid': int})
    df2 = pd.read_json(runfile2, lines=True, dtype={'q_num': str, 'qid': int})

    df = pd.merge(df1, df2, on=['q_num', 'qid'], suffixes=['_1', '_2'])

    print("Converting ranks...")
    df[['ranking_1_ranks', 'ranking_2_ranks']] = df.progress_apply(to_ranking, axis=1, result_type='expand')

    print("Computing KT...")
    df[['kt_tau', 'kt_p']] = df.progress_apply(
        lambda row: scipy.stats.kendalltau(row.ranking_1_ranks, row.ranking_2_ranks), axis=1, result_type='expand')
    return df


def main():
    parser = argparse.ArgumentParser(description='compare runs with kendalls tau correlation')
    parser.add_argument('-b', '--file1', dest='basefile', help='first file with runs')
    parser.add_argument('-c', '--file2', dest='compfile', help='second file with runs')
    args = parser.parse_args()

    basefile = args.basefile
    compfile = args.compfile
    assert compare_items_in_rankings(basefile, compfile)

    ktdf = kendalls_taus(basefile, compfile)
    print(ktdf.kt_tau.mean())


if __name__ == '__main__':
    main()
