import os
from collections import Counter
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import fairsearchdeltr

from features.features import AnnotationFeatureEngineer
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler
from reranker.deltr import Deltr


class TestingDeltr(Deltr):
    def __init__(self, pr, num_iter, gamma=1, random_state=None):
        gamma = gamma  # value of the gamma parameter
        number_of_iterations = num_iter  # number of iterations the training should run

        self.rs = random_state
        self.dtr = fairsearchdeltr.Deltr(pr, gamma, number_of_iterations, standardize=True)

    def train(self, data):
        np.random.seed(self.rs)
        return self.dtr.train(data)


@pytest.fixture
def root():
    return '/mnt/c/Users/maaik/Documents/thesis'


@pytest.fixture
def ft(root):
    esf = os.path.join(root, 'src/features/es-features-ferraro-sample-2020.csv')

    ft = AnnotationFeatureEngineer(doc_annotations=os.path.join(root, 'src/features/doc-annotations.csv'),
                                   es_feature_mat=esf)
    return ft


@pytest.fixture
def doclevel_map():
    return {'L': 1, 'Mixed': 1, 'H': 0}


def test_gamma_zero_notis_gamma_one_exposure_on_non_protected():
    num_iter = 10
    pr = 'gender'
    gamma1 = 0
    gamma2 = 1
    rs = 0
    deltr1 = fairsearchdeltr.Deltr(pr, gamma1, num_iter, standardize=True)

    deltr2 = fairsearchdeltr.Deltr(pr, gamma2, num_iter, standardize=True)

    train_data_raw = """q_id,doc_id,gender,score,judgment
                  1,1,0,0.962650646167003,1
                  1,2,1,0.940172822166108,0.1
                  """
    train_data = pd.read_csv(StringIO(train_data_raw))

    np.random.seed(rs)
    w1 = deltr1.train(train_data)

    np.random.seed(rs)
    w2 = deltr2.train(train_data)
    assert (w1 != w2).all()


def test_gamma_zero_is_gamma_one_exposure_on_protected():
    """
    When the exposure is on the protected group, for a small sample like this there is no difference between gamma zero
    and gamma one because if there is more exposure on the prot group the difference is reset to zero.
    This means that the protected group is allowed to get more exp than the non-protected group.
    :return:
    """
    num_iter = 10
    pr = 'gender'
    gamma1 = 0
    gamma2 = 1
    rs = 0
    deltr1 = fairsearchdeltr.Deltr(pr, gamma1, num_iter, standardize=True)

    deltr2 = fairsearchdeltr.Deltr(pr, gamma2, num_iter, standardize=True)

    train_data_raw = """q_id,doc_id,gender,score,judgment
                  1,1,1,0.962650646167003,1
                  1,2,0,0.940172822166108,0.1
                  """
    train_data = pd.read_csv(StringIO(train_data_raw))

    np.random.seed(rs)
    w1 = deltr1.train(train_data)

    np.random.seed(rs)
    w2 = deltr2.train(train_data)
    assert (w1 == w2).all()


def test_deltr_wrapper_same_outcome_as_testing_straight_deltr():
    num_iter = 10
    pr = 'gender'
    gamma = 1
    rs = 0
    deltr = TestingDeltr(pr, num_iter, gamma=gamma, random_state=rs)
    fsdeltr = fairsearchdeltr.Deltr(pr, gamma, num_iter, standardize=True)

    train_data_raw = """q_id,doc_id,gender,score,judgment
               1,1,1,0.962650646167003,1
               1,2,0,0.940172822166108,0.98
               1,3,0,0.925288002880488,0.96
               1,4,1,0.896143226020877,0.94
               1,5,0,0.89180775633204,0.92
               1,6,0,0.838704766545679,0.9
               """
    train_data = pd.read_csv(StringIO(train_data_raw))

    w1 = deltr.train(train_data)
    np.random.seed(rs)
    w2 = fsdeltr.train(train_data)
    assert (w1 == w2).all()


def test_deltr_wrapper_same_training_weights_as_fairsearch_deltr(root, ft, doclevel_map):
    num_iter = 10
    gamma = 1
    rs = 0
    q = os.path.join(root, 'training/2020/TREC-Fair-Ranking-training-sample.json')
    s = 'seq-test-train-2020-double-first-double-second.tsv'
    pr = 'DocHLevel'

    deltr = Deltr(ft, pr, doclevel_map, gamma, num_iter, random_state=rs)
    fsdeltr = fairsearchdeltr.Deltr("protected", gamma, num_iter, standardize=True)

    ioh = InputOutputHandler(s, q)
    features = deltr._prepare_data(ioh)

    w1 = deltr.train(ioh)
    np.random.seed(rs)
    w2 = fsdeltr.train(features)
    assert (w1 == w2).all()


def test_deltr_wrapper_two_rankers_one_gamma_same_training_weights(root, ft, doclevel_map):
    num_iter = 10
    gamma1 = 1
    gamma2 = gamma1
    rs = 0
    qt = os.path.join(root, 'training/2020/TREC-Fair-Ranking-training-sample.json')
    st = 'seq-test-train-2020-double-first-double-second.tsv'

    pr = 'DocHLevel'

    deltr = Deltr(ft, pr, doclevel_map, gamma1, num_iter, random_state=rs, gamma2=gamma2, alpha=0.5)

    ioht = InputOutputHandler(st, qt)

    w1, w2 = deltr.train(ioht)

    assert (w1 == w2).all()


class TestingDoubleDeltr(Deltr):
    def _prepare_data(self, data, has_judgment=False):
        return data


def test_deltr_wrapper_two_rankers_two_gammas_different_training_weights(root, ft, doclevel_map):
    num_iter = 10
    gamma1 = 0
    gamma2 = 1
    rs = 0
    qt = 'sample-test-2020-train-flipped-rels-for-deltr.json'
    st = 'seq-test-train-2020-double-first-double-second.tsv'

    train_data_raw = """q_id,doc_id,protected,score,judgment
                      1,1,0,0.962650646167003,1
                      1,2,1,0.940172822166108,0.1
                      """
    train_data = pd.read_csv(StringIO(train_data_raw))

    deltr = TestingDoubleDeltr(ft, "gender", doclevel_map, gamma1, num_iter, random_state=rs, gamma2=gamma2, alpha=0.5)

    w1, w2 = deltr.train(train_data)

    assert (w1 != w2).all()


def test_deltr_wrapper_two_rankers_two_gammas_half_alpha_merge_rankings(root, ft, doclevel_map):
    num_iter = 10
    gamma1 = 0
    gamma2 = 1
    rs = 0

    train_data_raw = """q_id,doc_id,protected,score,judgment
                         1,1,0,0.962650646167003,1
                         1,2,1,0.940172822166108,0.1
                         """
    train_data = pd.read_csv(StringIO(train_data_raw))

    prediction_data_raw = """sid,q_num,doc_id,protected,score
            0,1,7,0,0.9645
            0,1,8,0,0.9524
            0,1,9,0,0.9285
            0,1,10,0,0.8961
            0,1,11,1,0.8911
            0,1,12,1,0.8312
            """
    prediction_data = pd.read_csv(StringIO(prediction_data_raw))

    deltr = TestingDoubleDeltr(ft, "gender", doclevel_map, gamma1, num_iter, random_state=rs, gamma2=gamma2, alpha=0.5)

    w1, w2 = deltr.train(train_data)
    deltr.predict(prediction_data)

    assert (w1 != w2).all()


def test_deltr_wrapper_same_prediction_as_fairsearch_deltr(root, ft, doclevel_map):
    num_iter = 10
    gamma = 1
    rs = 0
    qt = os.path.join(root, 'training/2020/TREC-Fair-Ranking-training-sample.json')
    st = 'seq-test-train-2020-double-first-double-second.tsv'
    qe = os.path.join(root, 'evaluation/2020/TREC-Fair-Ranking-eval-sample.json')
    se = 'seq-test-eval-2020-double-first-double-second.tsv'

    pr = 'DocHLevel'

    deltr = Deltr(ft, pr, doclevel_map, gamma, num_iter, random_state=rs)
    fsdeltr = fairsearchdeltr.Deltr("protected", gamma, num_iter, standardize=True)

    ioht = InputOutputHandler(st, qt)
    featt = deltr._prepare_data(ioht)

    iohe = InputOutputHandler(se, qe)
    feate = deltr._prepare_data(iohe, has_judgment=False)

    deltr.train(ioht)

    np.random.seed(rs)
    fsdeltr.train(featt)

    preds = deltr.predict(iohe)

    preds = preds.dropna()

    for sid in [0, 1]:
        for q_num in [0, 1]:
            out = fsdeltr.rank(feate[(feate.sid == sid) & (feate.q_num == q_num)][feate.columns[1:]])
            assert preds[(preds.sid == sid) & (preds.q_num == q_num)].sort_values(
                by='rank').doc_id.to_list() == out.doc_id.to_list()


def test_fairsearchdeltr_reproducible_with_seed():
    # load some train data (this is just a sample - more is better)
    np.random.seed(0)
    train_data_raw = """q_id,doc_id,gender,score,judgment
            1,1,1,0.962650646167003,1
            1,2,0,0.940172822166108,0.98
            1,3,0,0.925288002880488,0.96
            1,4,1,0.896143226020877,0.94
            1,5,0,0.89180775633204,0.92
            1,6,0,0.838704766545679,0.9
            """
    train_data = pd.read_csv(StringIO(train_data_raw))

    # setup the DELTR object
    protected_feature = "gender"  # column name of the protected attribute (index after query and document id)
    gamma = 1  # value of the gamma parameter
    number_of_iterations = 10  # number of iterations the training should run
    standardize = True  # let's apply standardization to the features

    # create the Deltr object
    dtr = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)

    # train the model
    weights = dtr.train(train_data)

    np.random.seed(0)
    dtr2 = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)
    weights2 = dtr2.train(train_data)

    assert (weights == weights2).all()


def test_fairsearchdeltr_different_results_different_seeds():
    # load some train data (this is just a sample - more is better)
    np.random.seed(0)
    train_data_raw = """q_id,doc_id,gender,score,judgment
            1,1,1,0.962650646167003,1
            1,2,0,0.940172822166108,0.98
            1,3,0,0.925288002880488,0.96
            1,4,1,0.896143226020877,0.94
            1,5,0,0.89180775633204,0.92
            1,6,0,0.838704766545679,0.9
            """
    train_data = pd.read_csv(StringIO(train_data_raw))

    # setup the DELTR object
    protected_feature = "gender"  # column name of the protected attribute (index after query and document id)
    gamma = 1  # value of the gamma parameter
    number_of_iterations = 10  # number of iterations the training should run
    standardize = True  # let's apply standardization to the features

    # create the Deltr object
    dtr = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)

    # train the model
    weights = dtr.train(train_data)

    np.random.seed(1)
    dtr2 = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)
    weights2 = dtr2.train(train_data)

    assert (weights != weights2).all()


def test_fairsearchdeltr_not_reproducible_when_seed_not_reset():
    train_data_raw = """q_id,doc_id,gender,score,judgment
                1,1,1,0.962650646167003,1
                1,2,0,0.940172822166108,0.98
                1,3,0,0.925288002880488,0.96
                1,4,1,0.896143226020877,0.94
                1,5,0,0.89180775633204,0.92
                1,6,0,0.838704766545679,0.9
                """
    train_data = pd.read_csv(StringIO(train_data_raw))

    # setup the DELTR object
    protected_feature = "gender"  # column name of the protected attribute (index after query and document id)
    gamma = 1  # value of the gamma parameter
    number_of_iterations = 10  # number of iterations the training should run
    standardize = True  # let's apply standardization to the features

    # create the Deltr object
    dtr = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)

    # train the model
    weights = dtr.train(train_data)

    dtr2 = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)
    weights2 = dtr2.train(train_data)

    assert (weights != weights2).all()


def test_prepare_data_correctly_adds_h_class(doclevel_map):
    root = '/mnt/c/Users/maaik/Documents/thesis'
    qtrain = os.path.join(root, 'training/2020/TREC-Fair-Ranking-training-sample.json')
    seqtrain = os.path.join(root, 'training/2020/training-sequence-full.tsv')

    qeval = os.path.join(root, 'evaluation/2020/TREC-Fair-Ranking-eval-sample.json')
    seqeval = os.path.join(root, 'evaluation/2020/TREC-Fair-Ranking-eval-seq.tsv')

    esf = os.path.join(root, 'src/features/es-features-ferraro-sample-2020.csv')

    ft = AnnotationFeatureEngineer(doc_annotations=os.path.join(root, 'src/features/doc-annotations.csv'),
                                   es_feature_mat=esf)

    ioh_train = InputOutputHandler(seqtrain, qtrain)
    ioh_eval = InputOutputHandler(seqeval, qeval)

    deltr = Deltr(ft, 'DocHLevel', doclevel_map, gamma1=1, num_iter=None)
    feats_train = deltr._prepare_data(ioh_train)
    feats_eval = deltr._prepare_data(ioh_eval, has_judgment=False)

    assert len(feats_train) == 2049
    assert len(feats_eval) == 2161 * 150

    assert feats_train.columns.to_list() == ["q_num", "doc_id", "protected",
                                             "title_score", "abstract_score", "entities_score",
                                             "venue_score", "journal_score", "authors_score", "inCitations",
                                             "outCitations",
                                             "qlength",
                                             "relevance"]
    assert feats_eval.columns.to_list() == ["sid", "q_num", "doc_id", "protected",
                                            "title_score", "abstract_score", "entities_score",
                                            "venue_score", "journal_score", "authors_score", "inCitations",
                                            "outCitations",
                                            "qlength"]

    assert set(feats_train.protected.to_list()) == {0, 1}
    assert set(feats_eval.protected.to_list()) == {0, 1}

    traincounts = Counter(feats_train.protected.to_list())
    assert traincounts[1] == 1275
    assert traincounts[0] == 774

    evalcounts = Counter(feats_eval.protected.to_list())
    assert evalcounts[1] == 1415 * 150
    assert evalcounts[0] == 746 * 150


def test_fairsearchdeltr_github_example_works():
    np.random.seed(0)
    # load some train data (this is just a sample - more is better)
    train_data_raw = """q_id,doc_id,gender,score,judgment
        1,1,1,0.962650646167003,1
        1,2,0,0.940172822166108,0.98
        1,3,0,0.925288002880488,0.96
        1,4,1,0.896143226020877,0.94
        1,5,0,0.89180775633204,0.92
        1,6,0,0.838704766545679,0.9
        """
    train_data = pd.read_csv(StringIO(train_data_raw))

    # setup the DELTR object
    protected_feature = "gender"  # column name of the protected attribute (index after query and document id)
    gamma = 1  # value of the gamma parameter
    number_of_iterations = 10000  # number of iterations the training should run
    standardize = True  # let's apply standardization to the features

    # create the Deltr object
    dtr = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)

    # train the model
    weights = dtr.train(train_data)
    refweights = [0.02527054, 0.07692437]
    assert np.array([(refweights[i] - 0.015 <= weights[i]) & (refweights[i] + 0.015 >= weights[i]) for i in
                     range(0, len(weights))]).all()

    prediction_data_raw = """q_id,doc_id,gender,score
    1,7,0,0.9645
    1,8,0,0.9524
    1,9,0,0.9285
    1,10,0,0.8961
    1,11,1,0.8911
    1,12,1,0.8312
    """
    prediction_data = pd.read_csv(StringIO(prediction_data_raw))

    # use the model to rank the data
    res = dtr.rank(prediction_data)
    assert res.doc_id.to_list() == [11, 12, 7, 8, 9, 10]


def test_fairsearchdeltr_eval_data_with_missing_groups():
    np.random.seed(0)
    # load some train data (this is just a sample - more is better)
    train_data_raw = """q_id,doc_id,gender,score,judgment
        1,1,1,0.962650646167003,1
        1,2,0,0.940172822166108,0.98
        1,3,0,0.925288002880488,0.96
        1,4,1,0.896143226020877,0.94
        1,5,0,0.89180775633204,0.92
        1,6,0,0.838704766545679,0.9
        """
    train_data = pd.read_csv(StringIO(train_data_raw))

    # setup the DELTR object
    protected_feature = "gender"  # column name of the protected attribute (index after query and document id)
    gamma = 1  # value of the gamma parameter
    number_of_iterations = 10  # number of iterations the training should run
    standardize = True  # let's apply standardization to the features

    # create the Deltr object
    dtr = fairsearchdeltr.Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)

    # train the model
    dtr.train(train_data)

    prediction_data_raw = """q_id,doc_id,gender,score
    1,7,,0.9645
    1,8,0,0.9524
    1,9,,0.9285
    1,10,0,0.8961
    1,11,,0.8911
    1,12,1,0.8312
    """
    prediction_data = pd.read_csv(StringIO(prediction_data_raw))

    # use the model to rank the data
    res = dtr.rank(prediction_data)
    assert res.doc_id.to_list() == [11, 12, 7, 8, 9, 10]
