import pandas as pd
import pytest

from bonart.interface.iohandler import InputOutputHandler


class TestingInputOutputHandler(InputOutputHandler):
    def __init__(self):
        pass

    def read_sequence(self, fsequence):
        return self._InputOutputHandler__read_sequence(fsequence)


def read_sequence_og(fsequence):
    df = pd.read_csv(fsequence, names=["sid", "q_num", "qid"], sep='\.|,', engine='python')
    return df


@pytest.fixture
def dot_sequence():
    return 'dot_sequence.csv'


@pytest.fixture
def comma_sequence():
    return 'comma_sequence.csv'


@pytest.fixture
def hat_sequence():
    return 'hat_sequence.csv'



def test_read_sequence_dot_sequence():
    ioh = TestingInputOutputHandler()
    df_dot = ioh.read_sequence('dot_sequence.csv')
    df_com = ioh.read_sequence('comma_sequence.csv')
    df_hat = ioh.read_sequence('hat_sequence.csv')
    df_comp = pd.DataFrame({'sid': [0, 0, 0, 0, 0], 'q_num': [0, 1, 2, 3, 4], 'qid': [1, 2, 3, 4, 5]})

    assert (df_dot == df_comp).all().all()
    assert (df_com == df_comp).all().all()
    assert (df_hat == df_comp).all().all()

def test_updated_read_sequence_same_as_old():
    ioh = TestingInputOutputHandler()
    df_com = ioh.read_sequence('comma_sequence.csv')
    assert (df_com == read_sequence_og('comma_sequence.csv')).all().all()