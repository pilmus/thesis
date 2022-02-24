import pandas as pd
import pytest

from evaluation.kendall_tau import kendalls_taus, to_ranking


def test_to_ranking_same_runfile():
    rf1 = './runfile1.json'
    df1 = pd.read_json(rf1, lines=True, dtype={'q_num': str, 'qid': int})
    df2 = pd.read_json(rf1, lines=True, dtype={'q_num': str, 'qid': int})

    df = pd.merge(df1, df2, on=['q_num', 'qid'], suffixes=['_1', '_2'])
    l1, l2 = to_ranking(df.iloc[0])
    assert l1 == [0, 1, 2, 3, 4]
    assert l2 == [0, 1, 2, 3, 4]

def test_to_ranking_opposite_runfile():
    rf1 = './runfile1.json'
    rf2 = './runfile2.json'
    df1 = pd.read_json(rf1, lines=True, dtype={'q_num': str, 'qid': int})
    df2 = pd.read_json(rf2, lines=True, dtype={'q_num': str, 'qid': int})

    df = pd.merge(df1, df2, on=['q_num', 'qid'], suffixes=['_1', '_2'])
    l1, l2 = to_ranking(df.iloc[0])
    assert l1 == [0, 1, 2, 3, 4]
    assert l2 == [4, 3, 2, 1, 0]


def test_kendall_tau_same_runfile():
    rf1 = './runfile1.json'
    ktdf = kendalls_taus(rf1, rf1)
    assert round(ktdf.kt_tau.mean(), 1) == 1

def test_kendall_tau_opposite_runfile():
    rf1 = './runfile1.json'
    rf2 = './runfile2.json'
    ktdf = kendalls_taus(rf1, rf2)
    assert round(ktdf.kt_tau.mean(), 1) == -1
