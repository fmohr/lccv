import logging
import lccv
import numpy as np
import sklearn.datasets
import sklearn.tree
import unittest


class TestLccv(unittest.TestCase):

    def setUp(self):
        # setup logger for this test suite
        self.logger = logging.getLogger('lccv_test')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # configure lccv logger (by default set to WARN, change it to DEBUG if tests fail)
        self.lccv_logger = logging.getLogger("lccv")
        self.lccv_logger.setLevel(logging.WARN)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.lccv_logger.addHandler(ch)

    def test_partition_train_test_data(self):
        self.logger.info("Start Test on Partitioning")
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        for seed in [0, 1, 2, 3, 4, 5]:
            self.logger.info(f"Run test for seed {seed}")
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
            self.logger.info(f"Finished test for seed {seed}")

    def test_lccv_normal_function(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        _, _, res, _ = lccv.lccv(learner, features, labels, base=2, min_exp=4, enforce_all_anchor_evaluations=True, logger=self.lccv_logger)
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 135})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")

    def test_lccv_all_points_finish(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        _, _, res, _ = lccv.lccv(learner, features, labels, r=0.05, base=2, min_exp=4, enforce_all_anchor_evaluations=True, logger=self.lccv_logger)
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 135})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")
        
    def test_lccv_all_points_skipped(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        _, _, res, _ = lccv.lccv(learner, features, labels, r=0.05, base=2, min_exp=4, enforce_all_anchor_evaluations=False, logger=self.lccv_logger)
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 135})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")
        
    def test_lccv_pruning(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        _, _, res, _ = lccv.lccv(learner, features, labels, r=0.01, base=2, min_exp=4, logger=self.lccv_logger)
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 135})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")
            
