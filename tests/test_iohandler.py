import pandas as pd
import pytest

from interface.iohandler import InputOutputHandler


class TestingInputOutputHandler(InputOutputHandler):
    def __init__(self):
        pass

    def read_sequence(self, fsequence):
        return self._InputOutputHandler__read_sequence(fsequence)


def read_sequence_og(fsequence):
    df = pd.read_csv(fsequence, names=["sid", "q_num", "qid"], sep='\.|,', engine='python')
    return df


@pytest.fixture
def seq_dot():
    return 'seq_dot.csv'


@pytest.fixture
def seq_comma():
    return 'seq_comma.csv'


@pytest.fixture
def seq_hat():
    return 'seq_hat.csv'


def test_read_sequence_seq_dot():
    ioh = TestingInputOutputHandler()
    df_dot = ioh.read_sequence('seq_dot.csv')
    df_com = ioh.read_sequence('seq_comma.csv')
    df_hat = ioh.read_sequence('seq_hat.csv')
    df_comp = pd.DataFrame({'sid': [0, 0, 0, 0, 0], 'q_num': [0, 1, 2, 3, 4], 'qid': [1, 2, 3, 4, 5]})

    assert (df_dot == df_comp).all().all()
    assert (df_com == df_comp).all().all()
    assert (df_hat == df_comp).all().all()


def test_updated_read_sequence_same_as_old():
    ioh = TestingInputOutputHandler()
    df_com = ioh.read_sequence('seq_comma.csv')
    assert (df_com == read_sequence_og('seq_comma.csv')).all().all()
