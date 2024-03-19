import logging
import lccv
import numpy as np
import sklearn.datasets
from sklearn import *
import unittest
from parameterized import parameterized
import itertools as it
import time
import openml
import pandas as pd


def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid, download_data=True)
    df = ds.get_data()[0].dropna()
    y = df[ds.default_target_attribute].values
    X = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]]).values.astype(float)
    return X, y


class TestLccv(unittest.TestCase):
    
    preprocessors = [None]#, sklearn.preprocessing.RobustScaler, sklearn.kernel_approximation.RBFSampler]
    
    learners = [sklearn.tree.DecisionTreeClassifier]#sklearn.svm.LinearSVC, sklearn.tree.ExtraTreeClassifier, sklearn.linear_model.LogisticRegression, sklearn.linear_model.PassiveAggressiveClassifier, sklearn.linear_model.Perceptron, sklearn.linear_model.RidgeClassifier, sklearn.linear_model.SGDClassifier, sklearn.neural_network.MLPClassifier, sklearn.discriminant_analysis.LinearDiscriminantAnalysis, sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis, sklearn.naive_bayes.BernoulliNB, sklearn.naive_bayes.MultinomialNB, sklearn.neighbors.KNeighborsClassifier, sklearn.ensemble.ExtraTreesClassifier, sklearn.ensemble.RandomForestClassifier, sklearn.ensemble.GradientBoostingClassifier, sklearn.ensemble.GradientBoostingClassifier, sklearn.ensemble.HistGradientBoostingClassifier]

    @staticmethod
    def setUpClass():
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # setup logger for this test suite
        logger = logging.getLogger('lccv_test')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(ch)

        # configure lccv logger (by default set to WARN, change it to DEBUG if tests fail)
        lccv_logger = logging.getLogger("lccv")
        lccv_logger.setLevel(logging.INFO)
        lccv_logger.addHandler(ch)
        elm_logger = logging.getLogger("elm")
        elm_logger.setLevel(logging.WARN)
        elm_logger.addHandler(ch)
        
    def setUp(self):
        self.logger = logging.getLogger("lccv_test")
        self.lccv_logger = logging.getLogger("lccv")

    def test_partition_train_test_data(self):

        """
            tests whether the partitions created with the LCCV partition function are propert and reproducible.
        :return:
        """

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

            # check reproducibility for same seed
            f_tr2, l_tr2, f_te2, l_te2 = lccv._partition_train_test_data(
                features, labels, n_te, seed)
            np.testing.assert_array_equal(f_te, f_te2)
            np.testing.assert_array_equal(l_te, l_te2)
            np.testing.assert_array_equal(f_tr, f_tr2)
            np.testing.assert_array_equal(l_tr, l_tr2)
            self.logger.info(f"Finished test for seed {seed}")

    def test_lccv_normal_function(self):
        '''
                Just test whether the function
                    * runs through successfully,
                    * syntactically behaves well, and
                    * produces (syntactically) the desired results.
            '''
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        _, _, res, elm = lccv.lccv(
            learner,
            features,
            labels,
            r=0.0,
            base=2,
            min_exp=4,
            enforce_all_anchor_evaluations=True,
            logger=self.lccv_logger
        )
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 135})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.assertTrue(isinstance(elm.df, pd.DataFrame))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")
        
    def test_lccv_custom_evaluator(self):
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        
        # evaluator 1: Does not return a time
        # evaluator 2: Does return a runtime
        rs = np.random.RandomState(0)

        class CustomEvaluator:

            def __init__(self, fittime):
                self.fittime = fittime

            def __call__(self, learner_inst, anchor, timeout, base_scoring, additional_scorings, vals):
                results = {
                    "fittime": self.fittime
                }
                for scoring, _ in [base_scoring] + additional_scorings:
                    results.update({
                    f"score_train_{scoring}": 0.66 + rs.normal(scale=0.1),
                    f"scoretime_train_{scoring}": 0,
                    f"score_test_{scoring}": 0.66 + rs.normal(scale=0.1),
                    f"scoretime_test_{scoring}": 0
                })
                return results
        evaluator1 = CustomEvaluator(fittime=0)
        evaluator2 = CustomEvaluator(fittime=10**4)
        for i, evaluator in enumerate([evaluator1, evaluator2]):

            # first test in isolation without LCCV
            results = evaluator1(
                learner_inst=None,
                anchor=None,
                timeout=None,
                base_scoring=("accuracy", sklearn.metrics.get_scorer("accuracy")),
                additional_scorings=[],
                vals="test"
            )
            self.assertIsNotNone(results)

            self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
            _, _, res, elm = lccv.lccv(
                None,
                None,
                None,
                r=0.0,
                base=2,
                min_exp=4,
                enforce_all_anchor_evaluations=True,
                logger=self.lccv_logger,
                evaluator=evaluator,
                evaluator_kwargs={"vals": None},
                target_anchor=135,
                raise_exceptions=True
            )
            self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 135})
            for key, val in res.items():
                self.logger.info(f"Key: {key}, Val: {val}")
                self.assertFalse(np.isnan(val['conf'][0]))
                self.assertFalse(np.isnan(val['conf'][1]))
            self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")
            
            # check that registered runtime is correct
            mean_runtime = elm.df["fittime"].mean()
            if i == 0:
                self.assertTrue(mean_runtime < 10**-3)
            else:
                self.assertEqual(10**4, mean_runtime)
        
        # test scoring
        learner = sklearn.linear_model.LogisticRegression()
        features, labels = sklearn.datasets.load_digits(return_X_y=True)
        for scoring, is_binary in zip(["accuracy", "top_k_accuracy", "neg_log_loss", "roc_auc"], [False, True, False, True]):
            self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__} with scoring  {scoring}")

            y = (labels == 0 if is_binary else labels)

            _, _, res, elcm = lccv.lccv(
                learner,
                X=features,
                y=y,
                r=-np.inf,
                base=2,
                min_exp=2,
                enforce_all_anchor_evaluations=True,
                logger=self.lccv_logger,
                base_scoring=scoring,
                raise_exceptions=False, # small anchors will not work
                seed=3
            )
            self.assertSetEqual(set(res.keys()), {4, 8, 16, 32, 64, 128, 256, 512, 1024, 1617})
            for key, val in res.items():
                self.logger.info(f"Key: {key}, Val: {val}")
                if key > 16:
                    self.assertFalse(np.any(np.isnan(val['conf'])))
            self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")
            
            # check that registered runtime is correct
            mean_runtime = elm.df["fittime"].mean()
            expected_runtime = 0 if i == 0 else 10**4
            self.assertEqual(expected_runtime, mean_runtime)
            
            # check that results are negative for neg_log_loss and positive for accuracy
            means = np.array([e["mean"] for size, e in res.items() if size > 16])
            if scoring == "neg_log_loss":
                means *= -1
            self.assertTrue(np.all(means >= 0), f"Means for {scoring} should be non-negative but are {means}")
            
    def test_custom_schedule(self):
        """
            test that everything works fine if one provide a custom schedule over anchors
        :return:
        """
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test with custom schedule of LCCV on {learner.__class__.__name__}")
        _, _, _, elm = lccv.lccv(
            learner,
            features,
            labels,
            r=0.95,
            schedule=[10, 20],
            enforce_all_anchor_evaluations=False,
            logger=self.lccv_logger,
            seed=12
        )
        train_sizes = elm.df["anchor"].values
        self.assertEqual(10, train_sizes[0])
        self.assertEqual(2, len(np.unique(train_sizes)))
        self.assertEqual(20, np.max(train_sizes))
        with self.assertRaises(ValueError):
            lccv.lccv(learner, features, labels, r=0.95, schedule=[20, 10], enforce_all_anchor_evaluations=False, logger=self.lccv_logger, seed = 12)
        self.logger.info(f"Finished test of LCCV with custom schedule on {learner.__class__.__name__}")

    def test_lccv_all_points_finish(self):
        features, labels = sklearn.datasets.load_iris(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        _, _, res, _ = lccv.lccv(learner, features, labels, r=0.95, base=2, min_exp=4, enforce_all_anchor_evaluations=True, logger=self.lccv_logger)
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
        _, _, res, _ = lccv.lccv(learner, features, labels, r=0.95, base=2, min_exp=4, enforce_all_anchor_evaluations=False, logger=self.lccv_logger, seed = 12)
        self.assertSetEqual(set(res.keys()), {16, 135})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")

    def test_that_lccv_does_not_skip_immediately_but_eventually_to_highest_anchor(self):

        X, y = get_dataset(737)

        # the logic should work independently of whether train data is used or not since this is only for pruning
        for use_train in [True, False]:
            learners = [
                sklearn.tree.DecisionTreeClassifier(),
                sklearn.ensemble.RandomForestClassifier(random_state=42)
            ]

            r = -np.inf
            for i, learner in enumerate(learners):
                score, _, res, _ = lccv.lccv(
                    learner,
                    X,
                    y,
                    r=r,
                    base=2,
                    min_exp=4,
                    logger=self.lccv_logger,
                    use_train_curve=use_train
                )
                if score >= r:
                    r = np.max([score, r])

                if i == 1:  # RF should jump early
                    self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 2796})
                    for key, val in res.items():
                        self.logger.info(f"Key: {key}, Val: {val}")
                        self.assertFalse(np.isnan(val['conf'][0]))
                        self.assertFalse(np.isnan(val['conf'][1]))

    def test_lccv_pruning_in_real_case(self):

        X, y = get_dataset(737)
        learners = [
            sklearn.tree.DecisionTreeClassifier(random_state=42),
            sklearn.dummy.DummyClassifier()
        ]

        r = -np.inf
        for i, learner in enumerate(learners):
            score, _, res, _ = lccv.lccv(
                learner,
                X,
                y,
                r=r,
                base=2,
                min_exp=4,
                logger=self.lccv_logger,
                use_train_curve=False
            )
            if score >= r:
                r = np.max([score, r])

            if i == 1:  # majority classifier should stop early
                self.assertTrue(
                    max(res.keys()) <= 1024,
                    f"Biggest anchor evaluated by majority classifier should be 1024 but is {max(res.keys())}"
                )
                for key, val in res.items():
                    self.logger.info(f"Key: {key}, Val: {val}")
                    self.assertFalse(np.isnan(val['conf'][0]))
                    self.assertFalse(np.isnan(val['conf'][1]))

            if i == 1:  # XT-Forest should jump early
                self.assertTrue(
                    max(res.keys()) <= 1024,
                    f"Biggest anchor evaluated by majority classifier should be 1024 but is {max(res.keys())}"
                )
                for key, val in res.items():
                    self.logger.info(f"Key: {key}, Val: {val}")
                    self.assertFalse(np.isnan(val['conf'][0]))
                    self.assertFalse(np.isnan(val['conf'][1]))

    def test_lccv_pruning_in_theoretic_cases(self):
        features, labels = sklearn.datasets.load_digits(return_X_y=True)
        learner = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.logger.info(f"Starting test of LCCV on {learner.__class__.__name__}")
        
        r = 2.0  # this is not reachable (in accuracy), so there should be a pruning
        
        _, _, res, _ = lccv.lccv(
            learner,
            features,
            labels,
            r=r,
            base=2,
            min_exp=4,
            enforce_all_anchor_evaluations=True,
            logger=self.lccv_logger,
            use_train_curve=False
        )
        self.assertSetEqual(set(res.keys()), {16, 32, 64, 128, 256})
        for key, val in res.items():
            self.logger.info(f"Key: {key}, Val: {val}")
            self.assertFalse(np.isnan(val['conf'][0]))
            self.assertFalse(np.isnan(val['conf'][1]))
        self.logger.info(f"Finished test of LCCV on {learner.__class__.__name__}")

    @parameterized.expand(list(
        it.product(
            preprocessors,
            learners,
            [3, 61, 737, 772, 1464, 1485],
            [5, 10],
            ["accuracy", "neg_log_loss"]
        )
    ))
    def test_that_lccv_is_comparable_to_mccv(self, preprocessor, learner, dataset, folds, scoring):
        """
                This test checks whether the results are equivalent to MCCV and runtime not much worse.
                LCCV must not be more than 50% slower than MCCV (theoretically not more than 100% slower).
        """
        X, y = get_dataset(dataset)
        r = -np.inf
        self.logger.info(f"Start Test LCCV when running with r={r} on dataset {dataset}")
        
        # configure pipeline
        steps = []
        if preprocessor is not None:
            pp = preprocessor()
            if "copy" in pp.get_params().keys():
                pp = preprocessor(copy=False)
            steps.append(("pp", pp))
        learner_inst = learner()

        # active warm starting if available, because this can cause problems, and we want to cover the case.
        if "warm_start" in learner_inst.get_params().keys():
            learner_inst = learner(warm_start=True)
        steps.append(("predictor", learner_inst))
        pl = sklearn.pipeline.Pipeline(steps)
        
        # do tests
        try:

            train_size = np.round(1 - 1 / folds, 2)
            self.logger.info(f"Running {folds}-MCCV for train size {train_size}")
            start = time.time()
            cv_result = sklearn.model_selection.cross_validate(
                sklearn.base.clone(pl),
                X,
                y,
                cv=sklearn.model_selection.StratifiedShuffleSplit(n_splits=folds, train_size=train_size),
                scoring=scoring
            )
            score_cv = np.mean(cv_result["test_score"])
            std_in_cv = np.std(cv_result["test_score"])
            end = time.time()
            runtime_cv = end - start
            self.logger.info(f"Finished {folds}CV within {round(runtime_cv, 2)}s with score {np.round(score_cv, 3)}.")

            # run LCCV
            lccv_name = f"{int(100 * train_size)}LCCV"
            self.logger.info(f"Running {lccv_name}")
            start = time.time()
            score_lccv, _, res, elm = lccv.lccv(
                sklearn.base.clone(pl),
                X,
                y,
                r=r,
                target_anchor=train_size,
                max_evaluations=folds,
                base_scoring=scoring
            )
            end = time.time()
            runtime_lccv = end - start
            self.logger.info(f"""
                Finished {lccv_name} within {round(runtime_lccv, 2)}s with score {np.round(score_lccv, 3)}.
                Runtime diff was {np.round(runtime_cv - runtime_lccv, 1)}s.
                Performance diff was {np.round(score_cv - score_lccv, 4)}.
                Standard Deviation observed in LCCV at highest anchor was {np.round(std_in_cv, 4)}."""
            )
            self.assertFalse(np.isnan(score_lccv))

            # check runtime and result
            tol = max([0.005 if scoring == "accuracy" else 0.1, 2 * std_in_cv])
            self.assertTrue(
                runtime_lccv <= max([runtime_cv + 1, runtime_cv * 1.5]),
                msg=f"""
                    Runtime of {lccv_name} was {runtime_lccv},
                     which is more than 1s and more than 50% more than the  {runtime_cv} runtime of {folds}CV."
                    "Pipeline was {pl} and dataset {dataset}"""
            )
            self.assertTrue(
                np.abs(score_cv - score_lccv) <= tol,
                msg=f"""
                    Avg Score ({scoring}) of {lccv_name} was {np.round(score_lccv, 4)},
                    which deviates by more than {np.round(tol, 5)} from the {np.round(score_cv, 4)} of {folds}CV.
                    Standard deviation by LCCV in last fold was {np.round(std_in_cv, 4)}.
                    Pipeline was {pl} and dataset {dataset}"""
            )
        except ValueError:
            print("Skipping case in which training is not possible!")

    @parameterized.expand(list(it.product(preprocessors, learners, [(61, 0.0), (1485, 0.2)])))
    def test_lccv_respects_timeouts(self, preprocessor, learner, dataset):
        """
            This checks whether LCCV respects the timeout
        """
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
            score_80lccv = lccv.lccv(sklearn.base.clone(pl), X, y, r=r, target_anchor=.8, max_evaluations=5, timeout=timeout)[0]
            end = time.time()
            runtime_80lccv = end - start
            self.assertTrue(runtime_80lccv <= timeout, msg=f"Permitted runtime exceeded. Permitted was {timeout}s but true runtime was {runtime_80lccv}")
        except ValueError:
            print("Skipping case in which training is not possible!")