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


def kendalls_taus(runfile1, runfile2):
    df1 = pd.read_json(runfile1, lines=True, dtype={'q_num': str, 'qid': int})
    df2 = pd.read_json(runfile2, lines=True, dtype={'q_num': str, 'qid': int})

    df = pd.merge(df1, df2, on=['q_num', 'qid'], suffixes=['_1', '_2'])
    tqdm.pandas()
    df['kt_tau'] = df.progress_apply(lambda row: scipy.stats.kendalltau(row.ranking_1, row.ranking_2)[0], axis=1)
    return df


# basefile = "resources/evaluation/2020/rawruns/trec_run/trec_run.Deltr-gammas.json"
# compfile = "resources/evaluation/2020/rawruns/deltr_gammas/deltr_gammas-alpha-0.0-corpus-2020-grouping-all_low.json"
#
# basefile = "resources/evaluation/2019/fairRuns/trec_run/fair_LambdaMART.json"
# compfile = "resources/evaluation/2019/fairRuns/lambdamart_bonart_semanticscholar2019_1000_cleaned_og_2.json"
# same = compare_items_in_rankings(basefile,
#                                  compfile)
# print(same)
#
# print('\t'.join(['min', 'max', 'mean']))
# # print('\t'.join([str(val) for val in find_ranklist_size(basefile)]))
#
# ktdf = kendalls_taus(basefile, compfile)
#
# ktdf


def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)


KendalltauResult = namedtuple('KendalltauResult', ('correlation', 'pvalue'))


def kendalltau(x, y, initial_lexsort=None, nan_policy='propagate', method='auto'):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))
    elif not x.size or not y.size:
        return KendalltauResult(np.nan, np.nan)  # Return NaN if arrays are empty

    # check both x and y
    cnx, npx = _contains_nan(x, nan_policy)
    cny, npy = _contains_nan(y, nan_policy)
    contains_nan = cnx or cny
    if npx == 'omit' or npy == 'omit':
        nan_policy = 'omit'

    if contains_nan and nan_policy == 'propagate':
        return KendalltauResult(np.nan, np.nan)

    elif contains_nan and nan_policy == 'omit':
        x = ma.masked_invalid(x)
        y = ma.masked_invalid(y)
        return mstats_basic.kendalltau(x, y, method=method)

    if initial_lexsort is not None:  # deprecate to drop!
        warnings.warn('"initial_lexsort" is gone!')

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
                (cnt * (cnt - 1.) * (cnt - 2)).sum(),
                (cnt * (cnt - 1.) * (2 * cnt + 5)).sum())

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        return KendalltauResult(np.nan, np.nan)

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    # Limit range to fix computational errors
    tau = min(1., max(-1., tau))

    if method == 'exact' and (xtie != 0 or ytie != 0):
        raise ValueError("Ties found, exact method cannot be used.")

    if method == 'auto':
        if (xtie == 0 and ytie == 0) and (size <= 33 or min(dis, tot - dis) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'

    if xtie == 0 and ytie == 0 and method == 'exact':
        # Exact p-value, see Maurice G. Kendall, "Rank Correlation Methods" (4th Edition), Charles Griffin & Co., 1970.
        c = min(dis, tot - dis)
        if size <= 0:
            raise ValueError
        elif c < 0 or 2 * c > size * (size - 1):
            raise ValueError
        elif size == 1:
            pvalue = 1.0
        elif size == 2:
            pvalue = 1.0
        elif c == 0:
            pvalue = 2.0 / math.factorial(size) if size < 171 else 0.0
        elif c == 1:
            pvalue = 2.0 / math.factorial(size - 1) if (size - 1) < 171 else 0.0
        else:
            new = [0.0] * (c + 1)
            new[0] = 1.0
            new[1] = 1.0
            for j in range(3, size + 1):
                old = new[:]
                for k in range(1, min(j, c + 1)):
                    new[k] += new[k - 1]
                for k in range(j, c + 1):
                    new[k] += new[k - 1] - old[k - j]
            pvalue = 2.0 * sum(new) / math.factorial(size) if size < 171 else 0.0

    elif method == 'asymptotic':
        # con_minus_dis is approx normally distributed with this variance [3]_
        var = (size * (size - 1) * (2. * size + 5) - x1 - y1) / 18. + (
                2. * xtie * ytie) / (size * (size - 1)) + x0 * y0 / (9. *
                                                                     size * (size - 1) * (size - 2))
        pvalue = special.erfc(np.abs(con_minus_dis) / np.sqrt(var) / np.sqrt(2))
    else:
        raise ValueError("Unknown method " + str(method) + " specified, please use auto, exact or asymptotic.")

    return KendalltauResult(tau, pvalue)


list1 = ['e849fbf4f6feefde8880b008a83299971fe575f7',
         '5a489e8d10da85de48d495d335095d4705f0475d',
         'a4bbfd5c8ddbe24f3d17c7d1b41210a95f156646',
         '88420ad7068d25e2ef1516cfb8e0379f22af8641',
         '4bb687898519238f94bb560641ccc48998478289',
         'f36108a701e833325379ffaf87b6c728930f3b2e']
list2 = ['5a489e8d10da85de48d495d335095d4705f0475d',
         'e849fbf4f6feefde8880b008a83299971fe575f7',
         '88420ad7068d25e2ef1516cfb8e0379f22af8641',
         'a4bbfd5c8ddbe24f3d17c7d1b41210a95f156646',
         'f36108a701e833325379ffaf87b6c728930f3b2e',
         '4bb687898519238f94bb560641ccc48998478289']
#
# print(kendalltau(list1, list2))
print(kendalltau([0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]))
#
df = pd.DataFrame({'r1': [list1], 'r2': [list2]})
# print(df)
# print(df.corr(method='kendall'))

print([list2.index(item) for item in list1])


def to_ranking(row):
    return list(range(0, len(row.r1))), [list2.index(item) for item in list1]


df [['r1_id', 'r2_id']] = df.apply(to_ranking, axis=1, result_type = 'expand')
print(df)

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
