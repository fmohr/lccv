import lccv
import numpy.testing
import sklearn.datasets
import unittest


class TestLccv(unittest.TestCase):

    def test_partition_train_test_data(self):

        for seed in range(5):
            n_tr = 32
            n_te = 32
            features, labels = sklearn.datasets.load_iris(return_X_y=True)
            f_tr, l_tr, f_te, l_te = lccv._partition_train_test_data(
                features, labels, n_tr, n_te, seed)
            # check correct sizes
            self.assertEqual(f_tr.shape, (n_tr, 4))
            self.assertEqual(l_tr.shape, (n_tr, ))
            self.assertEqual(f_te.shape, (n_te, 4))
            self.assertEqual(l_te.shape, (n_te, ))
            # assume exact same test set, also when train set is double
            f_tr2, l_tr2, f_te2, l_te2 = lccv._partition_train_test_data(
                features, labels, n_tr * 2, n_te, seed)
            numpy.testing.assert_array_equal(f_te, f_te2)
            numpy.testing.assert_array_equal(l_te, l_te2)
            self.assertEqual(f_tr2.shape, (n_tr * 2, 4))
            self.assertEqual(l_tr2.shape, (n_tr * 2, ))
