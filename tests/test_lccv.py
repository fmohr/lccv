import lccv
import numpy as np
import sklearn.datasets
import sklearn.tree
import unittest


class TestLccv(unittest.TestCase):

    def test_partition_train_test_data(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        for seed in [0, 1, 2, 3, 4, 5]:
            n_te = 32
            n_tr = 150 - n_te
            f_tr, l_tr, f_te, l_te = lccv._partition_train_test_data(
                features, labels, n_te, seed)
            # check correct sizes
            self.assertEqual(f_tr.shape, (n_tr, 4))
            self.assertEqual(l_tr.shape, (n_tr, ))
            self.assertEqual(f_te.shape, (n_te, 4))
            self.assertEqual(l_te.shape, (n_te, ))
            # assume exact same test set, also when train set is double
            f_tr2, l_tr2, f_te2, l_te2 = lccv._partition_train_test_data(
                features, labels, n_te, seed)
            np.testing.assert_array_equal(f_te, f_te2)
            np.testing.assert_array_equal(l_te, l_te2)
            np.testing.assert_array_equal(f_tr, f_tr2)
            np.testing.assert_array_equal(l_tr, l_tr2)

    def test_lccv_normal_function(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        _, _, res, _ = lccv.lccv(learner, features, labels, base=2, min_exp=4, enforce_all_anchor_evaluations=True)
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 135})
        for key, val in res.items():
            print(key, val)
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))

    def test_lccv_all_points_finish(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        _, _, res, _ = lccv.lccv(learner, features, labels, r=0.01, base=2, min_exp=4)
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 135})
        for key, val in res.items():
            print(key, val)
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))