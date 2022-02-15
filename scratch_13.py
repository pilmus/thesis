import math
import warnings

import numpy as np
from numpy import array, asarray, ma
import pandas as pd
import scipy.stats
from scipy import special
from scipy.stats import mstats_basic, rankdata
from scipy.stats._stats import _kendall_dis
from tqdm import tqdm
from collections import Counter, namedtuple


def compare_items_in_rankings(runfile1, runfile2):
    """Check if rankings contain same items for each qnum/qid combination."""
    df1 = pd.read_json(runfile1, lines=True, dtype={'q_num': str, 'qid': int})
    df2 = pd.read_json(runfile2, lines=True, dtype={'q_num': str, 'qid': int})

    df = pd.merge(df1, df2, on=['q_num', 'qid'], suffixes=['_1', '_2'])

    return len(df[df.apply(
        lambda row: not (set(row.ranking_1).issubset(row.ranking_2) & set(row.ranking_1).issubset(row.ranking_2)),
        axis=1)]) == 0

    # merge on qnum, qid
    # ranking 1 & ranking 2 get own name
    # df.apply(lambda row: set(row.ranking1).issubset(row.rankin2) && set(row.ranking2).issubset(row.ranking1), axis =1)


def find_ranklist_size(runfile):
    df = pd.read_json(runfile, lines=True, dtype={'q_num': str, 'qid': int})
    min = df.apply(lambda row: len(row.ranking), axis=1).min()
    max = df.apply(lambda row: len(row.ranking), axis=1).max()
    mean = df.apply(lambda row: len(row.ranking), axis=1).mean()

    return min, max, mean


def to_ranking(row):
    return list(range(0, len(row.ranking_1))), [row.ranking_1.index(item) for item in row.ranking_2]


tqdm.pandas()


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


# basefile = "resources/evaluation/2020/rawruns/trec_run/trec_run.Deltr-gammas.json"
# compfile = "resources/evaluation/2020/rawruns/deltr_gammas/deltr_gammas-alpha-0.0-corpus-2020-grouping-all_low.json"
#
basefile = "resources/evaluation/2019/fairRuns/trec_run/fair_LambdaMART.json"
compfile = "resources/evaluation/2019/fairRuns/lambdamart_bonart_semanticscholar2019_1000_cleaned_og_2.json"
same = compare_items_in_rankings(basefile,
                                 compfile)
print(same)
#
# print('\t'.join(['min', 'max', 'mean']))
# # print('\t'.join([str(val) for val in find_ranklist_size(basefile)]))
#
ktdf = kendalls_taus(basefile, compfile)
#
# ktdf


print(kendalls_taus(basefile, compfile).kt_tau.mean())
print(kendalls_taus(compfile, basefile).kt_tau.mean())
print(kendalls_taus("resources/evaluation/2020/rawruns/trec_run/trec_run.LM-relevance.json", "resources/evaluation/2020/rawruns/lambdamart_ferraro/lambdamart_semanticscholar2020_10000_seq_rev_False_2.json").kt_tau.mean())
print(kendalls_taus("resources/evaluation/2020/rawruns/trec_run/trec_run.Deltr-gammas.json","resources/evaluation/2020/rawruns/deltr_gammas/deltr_gammas-alpha-0.25-corpus-2020-grouping-all_low.json").kt_tau.mean())

#
# print(rankdata(np.array([list1,list2])))
# print(np.array([list1,list2]).transpose())
# rankings = (np.array([list1, list2]).transpose().argsort(axis=0).argsort(axis=0) + 1).transpose()
# print(rankings[0],rankings[1])
# print(kendalltau(rankings[0], rankings[1]))
#
#
# print(rankdata(np.array([list1,list2]).transpose(), axis=0,method='ordinal'))
#
# print(df.rank(axis=0))
#
