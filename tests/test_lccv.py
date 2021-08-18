import logging
import lccv
import numpy as np
import sklearn.datasets
from sklearn import *
import unittest
from parameterized import parameterized
import itertools as it
import time
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
import openml
import pandas as pd

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0].dropna()
    y = df[ds.default_target_attribute].values

    categorical_attributes = df.select_dtypes(exclude=['number']).columns
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= len(pd.unique(df[att]))
        if expansion_size > 10**5:
            break

    if expansion_size < 10**5:
        X = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]]).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], sparse=True)

        print("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
    return X, y
    
class TestLccv(unittest.TestCase):
    
    preprocessors = [None, sklearn.preprocessing.RobustScaler, sklearn.kernel_approximation.RBFSampler]
    
    learners = [sklearn.svm.LinearSVC, sklearn.tree.DecisionTreeClassifier, sklearn.tree.ExtraTreeClassifier, sklearn.linear_model.LogisticRegression, sklearn.linear_model.PassiveAggressiveClassifier, sklearn.linear_model.Perceptron, sklearn.linear_model.RidgeClassifier, sklearn.linear_model.SGDClassifier, sklearn.neural_network.MLPClassifier, sklearn.discriminant_analysis.LinearDiscriminantAnalysis, sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis, sklearn.naive_bayes.BernoulliNB, sklearn.naive_bayes.MultinomialNB, sklearn.neighbors.KNeighborsClassifier, sklearn.ensemble.ExtraTreesClassifier, sklearn.ensemble.RandomForestClassifier, sklearn.ensemble.GradientBoostingClassifier, sklearn.ensemble.GradientBoostingClassifier, sklearn.ensemble.HistGradientBoostingClassifier]

    def setUpClass():
        # setup logger for this test suite
        logger = logging.getLogger('lccv_test')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # configure lccv logger (by default set to WARN, change it to DEBUG if tests fail)
        lccv_logger = logging.getLogger("lccv")
        lccv_logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        lccv_logger.addHandler(ch)
        
    def setUp(self):
        self.logger = logging.getLogger("lccv_test")
        self.lccv_logger = logging.getLogger("lccv")

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
        features, labels = sklearn.datasets.load_digits(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        _, _, res, _ = lccv.lccv(learner, features, labels, r=-0.5, base=2, min_exp=4, enforce_all_anchor_evaluations=True, logger=self.lccv_logger)
        self.assertSetEqual(set(res.keys()), {16, 32, 64})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")
            

        
    """
        This test checks whether the results are equivalent to a 5CV or 10CV
    """
    @parameterized.expand(list(it.product(preprocessors, learners, [1464])))#[61])))
    def test_lccv_runtime_and_result_bare(self, preprocessor, learner, dataset):
        X, y = get_dataset(dataset)
        self.logger.info(f"Start Test LCCV when running with r=1.0 on dataset {dataset}")
        
        # configure pipeline
        steps = []
        if preprocessor is not None:
            pp = preprocessor()
            if "copy" in pp.get_params().keys():
                pp = preprocessor(copy=False)
            steps.append(("pp", pp))
        learner_inst = learner()
        if "warm_start" in learner_inst.get_params().keys(): # active warm starting if available, because this can cause problems.
            learner_inst = learner(warm_start=True)
        steps.append(("predictor", learner_inst))
        pl = sklearn.pipeline.Pipeline(steps)
        
        # do tests
        try:
            
            # run 5-fold CV
            self.logger.info("Running 5CV")
            start = time.time()
            score_5cv = 1 - np.mean(sklearn.model_selection.cross_validate(sklearn.base.clone(pl), X, y, cv=5)["test_score"])
            end = time.time()
            runtime_5cv = end - start
            self.logger.info(f"Finished 5CV within {runtime_5cv}s.")

            # run 80lccv
            self.logger.info("Running 80LCCV")
            start = time.time()
            score_80lccv = lccv.lccv(sklearn.base.clone(pl), X, y, target_anchor=.8, MAX_EVALUATIONS=5)[0]
            end = time.time()
            runtime_80lccv = end - start
            self.logger.info(f"Finished 80LCCV within {runtime_80lccv}s. Runtime diff was {np.round(runtime_5cv - runtime_80lccv, 1)}s. Performance diff was {np.round(score_5cv - score_80lccv, 2)}.")

            # check runtime and result
            tol = 0.1#0.05 if dataset != 61 else 0.1
            self.assertTrue(runtime_80lccv <= (runtime_5cv + 1), msg=f"Runtime of 80lccv was {runtime_80lccv}, which is more than the {runtime_5cv} of 5CV. Pipeline was {pl} and dataset {dataset}")
            self.assertTrue(np.abs(score_5cv - score_80lccv) <= tol, msg=f"Avg Score of 80lccv was {score_80lccv}, which deviates by more than {tol} from the {score_5cv} of 5CV. Pipeline was {pl} and dataset {dataset}")
            
            
            # run 10-fold CV
            self.logger.info("Running 10CV")
            start = time.time()
            score_10cv = 1 - np.mean(sklearn.model_selection.cross_validate(sklearn.base.clone(pl), X, y, cv=10)["test_score"])
            end = time.time()
            runtime_10cv = end - start
            self.logger.info(f"Finished 10CV within {runtime_10cv}s.")

            # run 90lccv
            self.logger.info("Running 90LCCV")
            start = time.time()
            score_90lccv = lccv.lccv(sklearn.base.clone(pl), X, y, target_anchor=.9)[0]
            end = time.time()
            runtime_90lccv = end - start
            self.logger.info(f"Finished 90LCCV within {runtime_90lccv}s. Runtime diff was {np.round(runtime_10cv - runtime_90lccv, 1)}s. Performance diff was {np.round(score_10cv - score_90lccv, 2)}.")

            # check runtime and result
            tol = .1# 0.05 if dataset != 61 else 0.1
            self.assertTrue(runtime_90lccv <= runtime_10cv + 1, msg=f"Runtime of 90lccv was {runtime_90lccv}, which is more than the {runtime_10cv} of 10CV. Pipeline was {pl} and dataset {dataset}")
            self.assertTrue(np.abs(score_10cv - score_90lccv) <= tol, msg=f"Avg Score of 90lccv was {score_90lccv}, which deviates by more than {tol} from the {score_10cv} of 10CV. Pipeline was {pl} and dataset {dataset}")
        
        except ValueError:
            print("Skipping case in which training is not possible!")
            
    """
        This test checks whether the results are equivalent to a 5CV or 10CV
    """
    @parameterized.expand(list(it.product(preprocessors, learners, [(61, 0.0), (1485, 0.2)])))
    def test_lccv_runtime_and_result_applied(self, preprocessor, learner, dataset):
        X, y = get_dataset(dataset[0])
        r = dataset[1]
        self.logger.info(f"Start Test LCCV when running with r={r} on dataset {dataset[0]} wither preprocessor {preprocessor} and learner {learner}")
        
        # configure pipeline
        steps = []
        if preprocessor is not None:
            pp = preprocessor()
            if "copy" in pp.get_params().keys():
                pp = preprocessor(copy=False)
            steps.append(("pp", pp))
        learner_inst = learner()
        if "warm_start" in learner_inst.get_params().keys(): # active warm starting if available, because this can cause problems.
            learner_inst = learner(warm_start=True)
        steps.append(("predictor", learner_inst))
        pl = sklearn.pipeline.Pipeline(steps)
        
        
        # do tests
        try:
            
            # run 5-fold CV
            self.logger.info("Running 5CV")
            start = time.time()
            score_5cv = 1 - np.mean(sklearn.model_selection.cross_validate(sklearn.base.clone(pl), X, y, cv=5)["test_score"])
            end = time.time()
            runtime_5cv = end - start
            self.logger.info(f"Finished 5CV within {round(runtime_5cv, 2)}s with score {np.round(score_5cv, 3)}.")
            
            # run 80lccv
            self.logger.info("Running 80LCCV")
            start = time.time()
            score_80lccv = lccv.lccv(sklearn.base.clone(pl), X, y, r=r, target_anchor=.8, MAX_EVALUATIONS=5)[0]
            end = time.time()
            runtime_80lccv = end - start
            self.logger.info(f"Finished 80LCCV within {round(runtime_80lccv, 2)}s with score {np.round(score_80lccv, 3)}. Runtime diff was {np.round(runtime_5cv - runtime_80lccv, 1)}s. Performance diff was {np.round(score_5cv - score_80lccv, 2)}.")

            # check runtime and result
            tol = 0.05 if dataset != 61 else 0.1
            self.assertTrue(runtime_80lccv <= 2 * (runtime_5cv + 1), msg=f"Runtime of 80lccv was {runtime_80lccv}, which is more than the {runtime_5cv} of 5CV. Pipeline was {pl}")
            if np.isnan(score_80lccv):
                self.assertTrue(score_5cv > r, msg=f"80lccv returned nan even though the {score_5cv} of 10CV is not worse than the threshold {r}. Pipeline was {pl} and dataset {dataset}")
            else:
                self.assertTrue(np.abs(score_5cv - score_80lccv) <= tol, msg=f"Avg Score of 80lccv was {score_80lccv}, which deviates by more than {tol} from the {score_5cv} of 5CV. Pipeline was {pl} and dataset {dataset}")
            
            
            # run 10-fold CV
            self.logger.info("Running 10CV")
            start = time.time()
            score_10cv = 1 - np.mean(sklearn.model_selection.cross_validate(sklearn.base.clone(pl), X, y, cv=10)["test_score"])
            end = time.time()
            runtime_10cv = end - start
            self.logger.info(f"Finished 10CV within {round(runtime_10cv, 2)}s with score {np.round(score_10cv, 3)}")

            # run 90lccv
            self.logger.info("Running 90LCCV")
            start = time.time()
            score_90lccv = lccv.lccv(sklearn.base.clone(pl), X, y, r=r, target_anchor=.9)[0]
            end = time.time()
            runtime_90lccv = end - start
            self.logger.info(f"Finished 90LCCV within {round(runtime_90lccv, 2)}s with score {np.round(score_90lccv, 3)}. Runtime diff was {np.round(runtime_10cv - runtime_90lccv, 1)}s. Performance diff was {np.round(score_10cv - score_90lccv, 2)}.")

            # check runtime and result
            tol = 0.05 if dataset != 61 else 0.1
            self.assertTrue(runtime_90lccv <= 2 * (runtime_10cv + 1), msg=f"Runtime of 90lccv was {runtime_90lccv}, which is more than the {runtime_10cv} of 10CV. Pipeline was {pl}")
            if np.isnan(score_90lccv):
                self.assertTrue(score_10cv > r, msg=f"90lccv returned nan even though the {score_10cv} of 10CV is not worse than the threshold {r}. Pipeline was {pl} and dataset {dataset}")
            else:
                self.assertTrue(np.abs(score_10cv - score_90lccv) <= tol, msg=f"Avg Score of 90lccv was {score_90lccv}, which deviates by more than {tol} from the {score_10cv} of 10CV. Pipeline was {pl} and dataset {dataset}")
        
        except ValueError:
            print("Skipping case in which training is not possible!")
            
            
    """
        This checks whether LCCV respects the timeout
    """
    @parameterized.expand(list(it.product(preprocessors, learners, [(61, 0.0), (1485, 0.2)])))
    def test_lccv_respects_timeouts(self, preprocessor, learner, dataset):
        X, y = get_dataset(dataset[0])
        r = dataset[1]
        self.logger.info(f"Start Test LCCV when running with r={r} on dataset {dataset[0]} wither preprocessor {preprocessor} and learner {learner}")
        
        # configure pipeline
        steps = []
        if preprocessor is not None:
            pp = preprocessor()
            if "copy" in pp.get_params().keys():
                pp = preprocessor(copy=False)
            steps.append(("pp", pp))
        learner_inst = learner()
        if "warm_start" in learner_inst.get_params().keys(): # active warm starting if available, because this can cause problems.
            learner_inst = learner(warm_start=True)
        steps.append(("predictor", learner_inst))
        pl = sklearn.pipeline.Pipeline(steps)
        
        timeout = 1.5
        
        # do tests
        try:
            
            # run 80lccv
            self.logger.info("Running 80LCCV")
            start = time.time()
            score_80lccv = lccv.lccv(sklearn.base.clone(pl), X, y, r=r, target_anchor=.8, MAX_EVALUATIONS=5, timeout=timeout)[0]
            end = time.time()
            runtime_80lccv = end - start
            self.assertTrue(runtime_80lccv <= timeout, msg=f"Permitted runtime exceeded. Permitted was {timeout}s but true runtime was {runtime_80lccv}")
        except ValueError:
            print("Skipping case in which training is not possible!")