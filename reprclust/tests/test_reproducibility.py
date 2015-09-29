from mock import patch
from mvpa2.datasets.base import Dataset
import numpy as np
from numpy.testing import assert_array_equal

from nose.tools import assert_is_none, assert_equal, assert_raises

from reprclust.cluster_methods import WardClusterMethod
from reprclust.reproducibility import reproducibility, _run_fold

from itertools import product

# create two far blobs easy to cluster
blob1 = 2*np.random.randn(10, 2) + 100
blob2 = 2*np.random.randn(10, 2) - 100
# 2 samples for 20 features
data = np.vstack((blob1, blob2)).T

# add some noise and transpose
dss = [(data + np.random.randn(*data.shape)) for i in xrange(10)]
# fake test, we should get a value of 1 for k=2
#idx_train = idx_test = range(10)
#fake_splitter = [(idx_train, idx_test)]
fake_splitter = product([range(10)], repeat=2)
ground_truth = np.hstack((np.zeros(10, dtype=int), np.ones(10, dtype=int)))

dss = np.vstack(dss)
dss = Dataset(dss, sa={'subjects': np.repeat(range(10), 2),
                       'runs': np.tile(range(2), 10)})

def test_run_method():
    scores = reproducibility(dss, fake_splitter, WardClusterMethod(),
                             ks=np.arange(2, 21),
                             ground_truth=ground_truth, verbose=0)
    for key, value in scores.items():
        assert_array_equal(value[0], np.arange(2, 21))
        if not key.endswith('_gt'):
            # this works only with ARI and AMI -- default
            assert_array_equal(value[1], np.ones(value[1].shape))
        else:
            # only the first is 1.
            assert_equal(value[1, 0], 1.)

def test_run_fold():
    common_args = (fake_splitter, WardClusterMethod(), [2, 3])
    # check raises
    assert_raises(TypeError, _run_fold, blob1, *common_args)
    assert_raises(KeyError, _run_fold, dss, *common_args, spaces=['sa.chunks'])

def test_nested_fold():
    splitter_subjects = [(range(5), range(5, 10))]
    splitter_runs = [([0], [1])]

    class FoldFxStore(object):
        def __init__(self):
            self.train = []
            self.test = []
            self.calls = 0
        def __call__(self, data_train, data_test):
            self.calls += 1
            self.train.append(data_train)
            self.test.append(data_test)
            return data_train.samples, data_test.samples

    mock_foldfx = FoldFxStore()

    scores = reproducibility(dss, [splitter_subjects, splitter_runs],
                             WardClusterMethod(),
                             spaces=['sa.subjects', 'sa.runs'],
                             fold_fx=mock_foldfx,
                             ks=[2],
                             ground_truth=ground_truth, verbose=0)

    assert_equal(mock_foldfx.calls, 1)
    assert_array_equal(np.unique(mock_foldfx.train[0].sa.subjects), splitter_subjects[0][0])
    assert_array_equal(np.unique(mock_foldfx.train[0].sa.runs), splitter_runs[0][0])
    assert_array_equal(np.unique(mock_foldfx.test[0].sa.subjects), splitter_subjects[0][1])
    assert_array_equal(np.unique(mock_foldfx.test[0].sa.runs), splitter_runs[0][1])

