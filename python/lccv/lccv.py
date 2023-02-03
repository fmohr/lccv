import typing
import logging

import numpy as np
import pandas as pd
import scipy.stats
import time
import sklearn.metrics
import func_timeout

import inspect

import matplotlib.pyplot as plt

def format_learner(learner):
    learner_name = str(learner).replace("\n", " ").replace("\t", " ")
    for k in  range(20):
        learner_name = learner_name.replace("  ", " ")
    return learner_name

def _partition_train_test_data(
        features: np.array, labels: np.array, n_test: int,
        seed: int) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    """
    Partitions the dataset in a test set of the size of the requested size, and
    a train set of size n_train.

    :param features: The X-data
    :param labels: The y-data
    :param n_test: the requested test size
    :param seed: The random seed
    :return: A 4-tuple, consisting of the train features (2D np.array), the
    train labels (1D np.array), the test features (2D np.array) and the test
    labels (1D np.array)
    """
    if seed is None:
        raise ValueError('Seed can not be None (to ensure test set equality)')
    np.random.seed(seed)
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    return features[n_test:], labels[n_test:], features[:n_test], labels[:n_test]


class EmpiricalLearningModel:
    
    def __init__(self, learner, X, y, n_target, seed, fix_train_test_folds, evaluator, scoring):
        
        # set up logger
        self.logger = logging.getLogger('elm')
        
        self.learner = learner
        self.active_seed = seed
        self.fix_train_test_folds = fix_train_test_folds
        
        self.scoring = scoring
        
        # set evaluator and scoring
        self.evaluator = evaluator if evaluator is not None else self.evaluate
        if not callable(self.evaluator):
            raise Exception(f"Evaluator is of type {type(self.evaluator)}, which is not a callable.")
        
        # the data is only used if no evaluator is given
        if evaluator is None:
            
            if X.shape[0] <= 0:
                raise Exception(f"Recieved dataset with non-positive number of instances. Shape is {X.shape}")
            
            n_test = X.shape[0] - n_target # portion of data that exceeds the target value is used for testing
            
            if fix_train_test_folds:
                self.X_train, self.y_train, self.X_test, self.y_test = _partition_train_test_data(X, y, n_test, seed)
                self.logger.info(f"Train labels: \n{self.y_train}")
                self.logger.info(f"Test labels: \n{self.y_test}")
            else:
                self.X = X
                self.y = y
                self.n_test = n_test
                
        # initialize data
        self.df = pd.DataFrame([], columns=["trainsize", "seed", "score_train", "score_test", "runtime"])
        self.rs = np.random.RandomState(seed)

    def evaluate(self, learner_inst, size, timeout):

        self.active_seed += 1
        self.logger.debug("Computing training data")
        
        # obtain train and test data (depending on configuration)
        if self.fix_train_test_folds:
            self.logger.info("Re-using pre-defined train and test folds")
            X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test
        else:
            X_train, y_train, X_test, y_test = _partition_train_test_data(self.X, self.y, self.n_test, self.active_seed)
            self.logger.info(f"Dynamically creating a train and test fold with seed {self.active_seed}.")
        
        indices = self.rs.choice(X_train.shape[0], size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        self.logger.debug(f"Created train portion. Labels in train/test data: {len(np.unique(y_train))}/{len(np.unique(y_test))}")
        
        hash_before = hash(X_train.tobytes())
        
        # if a scoring function is given as a string, the existing labels are added through make_scorer.
        # this is a work-around since sklearn does not allow to provide the labels when getting a scoring with get_scorer
        # it is also necessary here and NOT in the constructor, because the labels must be the ones used in the training set.
        if type(self.scoring) == str:
            tmp_scorer = sklearn.metrics.get_scorer(self.scoring)
            needs_labels = "labels" in inspect.signature(tmp_scorer._score_func).parameters
            kws = {
                "score_func": tmp_scorer._score_func,
                "greater_is_better": tmp_scorer._sign == 1,
                "needs_proba": type(tmp_scorer) == sklearn.metrics._scorer._ProbaScorer,
                "needs_threshold": type(tmp_scorer) == sklearn.metrics._scorer._ThresholdScorer,
            }
            if needs_labels:
                kws["labels"] = list(np.unique(y_train))
            scoring = sklearn.metrics.make_scorer(**kws)
        else:
            scoring = self.scoring
        if not callable(scoring):
            raise Exception(f"Scoring is of type {type(self.scoring)}, which is not a callable. Make sure to pass a string or Callable.")
        
        learner_inst = sklearn.base.clone(learner_inst)
        self.logger.info(f"Training {format_learner(learner_inst)} on data of shape {X_train.shape}. Timeout is {timeout}")
        start = time.time()
        if timeout is None:
            learner_inst.fit(X_train, y_train)
        else:
            func_timeout.func_timeout(timeout, learner_inst.fit, (X_train, y_train))
        end = time.time()
        self.logger.debug(f"Training ready after {int((end - start) * 1000)}ms. Now obtaining predictions.")
        score_test = scoring(learner_inst, X_test, y_test)
        score_train = scoring(learner_inst, X_train, y_train)
        end = time.time()
        self.logger.info(f"Evaluation ready after {int((end - start) * 1000)}ms. Score of model on {y_test.shape[0]} validation/test instances is {score_test}.")
        hash_after = hash(X_train.tobytes())
        if hash_before != hash_after:
            raise Exception("Evaluation of pipeline has changed the data. Please make sure to evaluate pipelines that do not change the data in place.")
        return score_train, score_test
    
    def compute_and_add_sample(self, size, seed=None, timeout=None, verbose=False):
        tic = time.time()
        # TODO: important to check whether this is always a different order
        evaluation_result = self.evaluator(
            self.learner, size,
            timeout / 1000 if timeout is not None else None)
        toc = time.time()
        runtime = int(np.round(1000 * (toc-tic)))
        
        # extract evaluation result (possibly overriding the runtime)
        if type(evaluation_result) != tuple:
            raise ValueError(f"Evaluator supposed to return a tuple but returned {type(evaluation_result)}")
        if len(evaluation_result) == 2:
            score_train, score_test = evaluation_result
        elif len(evaluation_result) == 3:
            score_train, score_test, runtime = evaluation_result
        else:
            raise ValueError(f"Evaluator returned a result of length {len(evaluation_result)} but must be 2 or 3.")
            
        self.logger.debug(f"Sample value computed within {runtime}ms")
        self.df.loc[len(self.df)] = [size, seed, score_train, score_test, runtime]
        self.df = self.df.astype({"trainsize": int, "seed": int, "runtime": int})
        return score_train, score_test
    
    def get_values_at_anchor(self, anchor, test_scores = True):
        return self.df[self.df["trainsize"] == anchor]["score_" + ("test" if test_scores else "train")].values
    
    def get_best_worst_train_score(self):
        return max([min(g) for i, g in self.df.groupby("trainsize")["score_train"]])
    
    def get_mean_performance_at_anchor(self, anchor, test_scores = True):
        return np.mean(self.get_values_at_anchor(anchor, test_scores = test_scores))
    
    def get_mean_curve(self, test_scores = True):
        anchors = sorted(pd.unique(self.df["trainsize"]))
        return anchors, [self.get_mean_performance_at_anchor(a, test_scores = test_scores) for a in anchors]
    
    def get_runtimes_at_anchor(self, anchor):
        return self.df[self.df["trainsize"] == anchor]["runtime"].values
    
    def get_conf_interval_size_at_target(self, target):
        if len (self.df[self.df["trainsize"] == target]) == 0:
            return 1
        ci = self.get_normal_estimates(size = target)["conf"]
        return ci[1] - ci[0]
    
    def get_lc_estimate_at_target(self, target):
        return self.get_mmf()[1](target)
    
    def get_normal_estimates(self, size = None, round_precision=100, validation = True):
        
        if size is None:
            sizes = sorted(np.unique(self.df["trainsize"]))
            out = {}
            for size in sizes:
                out[int(size)] = self.get_normal_estimates(size)
            return out
    
        dfProbesAtSize = self.df[self.df["trainsize"] == size]
        mu = np.mean(dfProbesAtSize["score_" + ("test" if validation else "train")])
        sigma = np.std(dfProbesAtSize["score_" + ("test" if validation else "train")])
        return {
            "n": len(dfProbesAtSize["score_" + ("test" if validation else "train")]),
            "mean": np.round(mu, round_precision),
            "std": np.round(sigma, round_precision),
            "conf": np.round(scipy.stats.norm.interval(0.95, loc=mu, scale=sigma/np.sqrt(len(dfProbesAtSize))) if sigma > 0 else (mu, mu), round_precision)
        }
    
    def get_slope_ranges(self):
        est = self.get_normal_estimates()
        sizes = [s for s in est]
        ranges = []
        for i, size in enumerate(sizes):
            if i > 0:
                anchor_size_prev_last = sizes[i - 1]
                anchor_size_last = sizes[i]
                
                # compute confidence bounds of prev last and last anchor
                if est[anchor_size_prev_last]["n"] > 1:
                    lower_prev_last = est[anchor_size_prev_last]["conf"][0]
                    upper_prev_last = est[anchor_size_prev_last]["conf"][1]
                else:
                    lower_prev_last = upper_prev_last = est[anchor_size_prev_last]["mean"]
                if est[anchor_size_last]["n"] > 1:
                    lower_last = est[anchor_size_last]["conf"][0]
                    upper_last = est[anchor_size_last]["conf"][1]
                else:
                    lower_last = upper_last = est[anchor_size_last]["mean"]
                
                # compute slope range
                pessimistic_slope = max(0, (lower_last - upper_prev_last) / (anchor_size_last - anchor_size_prev_last))
                optimistic_slope = max(0, (upper_last - lower_prev_last) / (anchor_size_last - anchor_size_prev_last))
                ranges.append((pessimistic_slope, optimistic_slope))
        return ranges
    
    def get_slope_range_in_last_segment(self):
        return self.get_slope_ranges()[-1]
    
    def get_performance_interval_at_target(self, target):
        pessimistic_slope, optimistic_slope = self.get_slope_range_in_last_segment()
        sizes = sorted(np.unique(self.df["trainsize"]))
        last_size = sizes[-1]
        normal_estimates = self.get_normal_estimates()[last_size]
        if normal_estimates["n"] > 1:
            last_conf = normal_estimates["conf"]
            if normal_estimates["std"] > 0:
                last_conf_lower = last_conf[0]
                last_conf_upper = last_conf[1]
            else:
                last_conf_lower = last_conf_upper = normal_estimates["mean"]
                last_conf = (last_conf_lower, last_conf_upper)
        else:
            last_conf_lower = last_conf_upper = normal_estimates["mean"]
            last_conf = (last_conf_lower, last_conf_upper)
        if any(np.isnan(last_conf)):
            raise Exception("Confidence interval must not be nan!")
        if np.isnan(optimistic_slope):
            raise Exception("Slope must not be nan")
        return pessimistic_slope * (target - last_size) + last_conf_lower, optimistic_slope * (target - last_size) + last_conf_upper
        
    def get_ipl(self):
        sizes = sorted(list(pd.unique(self.df["trainsize"])))
        scores = [np.mean(self.df[self.df["trainsize"] == s]["score_test"]) for s in sizes]
        def ipl(beta):
            a, b, c = tuple(beta.astype(float))
            pl = lambda x: a + b * x **(-c)
            penalty = []
            for i, size in enumerate(sizes):
                penalty.append((pl(size) - scores[i])**2)
            return np.array(penalty)

        a, b, c = tuple(scipy.optimize.least_squares(ipl, np.array([1,1,1]), method="lm").x)
        return lambda x: a + b * x **(-c)
    
    def get_mmf(self, validation_curve = True):
        sizes = sorted(list(pd.unique(self.df["trainsize"])))
        scores = [np.mean(self.df[self.df["trainsize"] == s]["score_" + ("test" if validation_curve else "train")]) for s in sizes]
        weights = [2**i for i in range(len(sizes))]
        def mmf(beta):
            a, b, c, d = tuple(beta.astype(float))
            fun = lambda x: (a * b + c * x ** d)/(b + x ** d)
            penalties = []
            for i, size in enumerate(sizes):
                penalty = weights[i]  * ((scores[i] - fun(size)) ** 2) # give more weights on higher anchors
                penalties.append(penalty if not np.isnan(penalty) else 10**6)
            return sum(penalties)
        
        factor = 1 if validation_curve else -1
        const = {
            "type": "ineq", "fun": lambda x: -factor * x[1] * (x[2]-x[0])*x[3],
            #"type": "ineq", "fun": lambda x: factor if all([(x[2] - x[0]) * ((x[3] + 1)* size**x[3] - x[1]*x[3] + x[2]) for size in np.linspace(64, 10000, 1000)]) else -factor
        }

        a, b, c, d = tuple(scipy.optimize.minimize(mmf, np.array([0.5,1,1,-1]), constraints=const).x)
        return (a, b, c, d), lambda x: (a * b + c * x ** d)/(b + x ** d)
    
    def predict_runtime(self, target_size):
        lr = sklearn.linear_model.LinearRegression()
        X = self.df[["trainsize"]].values
        X = np.row_stack([X, [[0]]])
        X = np.column_stack([X, X[:]**2])
        y = self.df["runtime"].values
        y = np.append(y, [0])
        lr.fit(X, y)
        b = np.abs(lr.coef_[0])
        a = np.abs(lr.coef_[1])
        return a * (target_size**2) + b * target_size + lr.intercept_
    
    def get_max_size_for_runtime(self, runtime):
        lr = sklearn.linear_model.LinearRegression()
        X = self.df[["trainsize"]].values
        X = np.row_stack([X, [[0]]])
        X = np.column_stack([X, X[:]**2])
        y = self.df["runtime"].values
        y = np.append(y, [0])
        lr.fit(X, y)
        b = np.abs(lr.coef_[0])
        a = np.abs(lr.coef_[1])
        inner = (-b/(2 * a))**2 - (lr.intercept_ - runtime) / a
        return -b/(2 * a) + np.sqrt(inner)
    
    def visualize(self, max_anchor = 1000, r = None):
        sizes = sorted(list(pd.unique(self.df["trainsize"])))
        scores_train = [self.get_normal_estimates(s, validation=False) for s in sizes]
        scores_valid = [self.get_normal_estimates(s, validation=True) for s in sizes]
        lc_train_params, lc_train = self.get_mmf(False)
        lc_test_params, lc_valid = self.get_mmf(True)
        
        fig, ax = plt.subplots()
        ax.scatter(sizes, [e["mean"] for e in scores_train])
        ax.scatter(sizes, [e["mean"] for e in scores_valid])
        domain = np.linspace(64, max_anchor, 100)
        ax.plot(domain, lc_train(domain), color="C0")
        ax.fill_between(sizes, [v["mean"] - v["std"] for v in scores_train], [v["mean"] + v["std"] for v in scores_train], alpha=0.2, color="C0")
        ax.plot(domain, lc_valid(domain), color="C1")
        ax.fill_between(sizes, [v["mean"] - v["std"] for v in scores_valid], [v["mean"] + v["std"] for v in scores_valid], alpha = 0.2, color="C1")
        
        # create lines that project based on convexity
        val_at_target_pessimistic, val_at_target_optimistic = self.get_performance_interval_at_target(max_anchor)
        ax.plot([sizes[-2], max_anchor], [scores_valid[-2]["mean"] + scores_valid[-2]["std"], val_at_target_pessimistic], color="C3", linestyle="--")
        ax.plot([sizes[-2], max_anchor], [scores_valid[-2]["mean"] - scores_valid[-2]["std"], val_at_target_optimistic], color="C2", linestyle="--")
        
        if r is not None:
            ax.axhline(r, color="black", linestyle="--")
        plt.show()
    

def lccv(learner_inst, X, y, r, timeout=None, base=2, min_exp=6, MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION=0.005, MAX_EVALUATIONS=10, target_anchor=.9, return_estimate_on_incomplete_runs=False, max_conf_interval_size_default=0.1, max_conf_interval_size_target=0.001, enforce_all_anchor_evaluations=False, seed=0, verbose=False, logger=None, min_evals_for_stability=3, use_train_curve=True,fix_train_test_folds=False, evaluator=None, scoring="accuracy", visualize_lcs = False, exceptions = "message"):
    """
    Evaluates a learner in an iterative fashion, using learning curves. The
    method builds upon the assumption that learning curves are convex. After
    each iteration, it checks whether the convexity assumption is still valid.
    If not, it tries to repair it.
    Also, after each iteration it checks whether the performance of the best
    seen learner so far is still reachable by making an optimistic extrapolation.
    If not, it stops the evaluation.

    :param learner_inst: The learner to be evaluated
    :param X: The features on which the learner needs to be evaluated
    :param y: The labels on which the learner needs to be trained
    :param r: The best seen performance so far (lower is better). Fill in 0.0 if
    no learners have been evaluated prior to the learner.
    :param timeout: The maximal runtime for this specific leaner. Fill in None
    to avoid cutting of the evaluation.
    :param base: The base factor to increase the sample sizes of the learning
    curve.
    :param min_exp: The first exponent of the learning curve.
    :param MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION: The maximum number of
    evaluations to be performed
    :param MAX_EVALUATIONS:
    :param target_anchor:
    :param return_estimate_on_incomplete_runs:
    :param max_conf_interval_size_default:
    :param max_conf_interval_size_target:
    :param enforce_all_anchor_evaluations:
    :param seed:
    :param verbose:
    :param logger:
    :param min_evals_for_stability:
    :param use_train_curve: If True, then the evaluation stops as soon as the train curve drops under the threshold r
    :param evaluator: Function to be used to query a noisy score at some anchor. To be maximized!
    :param scoring: Scoring function to be computed for predictions obtained at an anchor. Is ignored if an evaluator is given.
    :return:
    """
    # create standard logger if none is given
    if logger is None:
        logger = logging.getLogger('lccv')
    logger.debug("timeout = " + str(timeout) + ", " +
                 "BASE = " + str(base) + ", " +
                 "min_exp = " + str(min_exp) + ", " +
                 "MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION = " + str(MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION) + ", " +
                 "MAX_EVALUATIONS = " + str(MAX_EVALUATIONS) + ", " +
                 "target_anchor = " + str(target_anchor) + ", " +
                 "return_estimate_on_incomplete_runs = " + str(return_estimate_on_incomplete_runs) + ", " +
                 "max_conf_interval_size_default = " + str(max_conf_interval_size_default) + ", " +
                 "max_conf_interval_size_target = " + str(max_conf_interval_size_target) +  ", " +
                 "enforce_all_anchor_evaluations = " + str(enforce_all_anchor_evaluations) +  ", " +
                 "seed = " + str(seed) +  ", " +
                 "min_evals_for_stability = " + str(min_evals_for_stability) + ", " +
                 "fix_train_test_folds = " + str(fix_train_test_folds))
    # intialize
    tic = time.time()
    deadline = tic + timeout if timeout is not None else None

    # configure the exponents and status variables
    if target_anchor < 1:
        if X is None:
            raise Exception("If no data is given, the `target_anchor` parameter must be specified as a positive integer.")
        target_anchor = int(np.floor(X.shape[0] * target_anchor))
    
    # initialize important variables and datastructures
    max_exp = np.log(target_anchor) / np.log(base)
    schedule = [base**i for i in list(range(min_exp, int(np.ceil(max_exp))))] + [target_anchor]
    slopes = (len(schedule) - 1) * [np.nan]
    elm = EmpiricalLearningModel(learner_inst, X, y, target_anchor, seed, fix_train_test_folds, evaluator = evaluator, scoring = scoring)
    T = len(schedule) - 1
    t = 0 if r < np.inf or enforce_all_anchor_evaluations else T
    repair_convexity = False
    
    # announce start event together with state variable values
    logger.info(f"""Running LCCV {'on ' + str(X.shape) + '-shaped data' if X is not None else 'with custom evaluator.'}. Overview:
    learner: {format_learner(learner_inst)}
    r: {r}
    min_exp: {min_exp}
    max_exp: {max_exp}
    Seed is {seed}
    t_0: {t}
    Schedule: {schedule}""")
    
    ## MAIN LOOP
    while t <= T and elm.get_conf_interval_size_at_target(target_anchor) > max_conf_interval_size_target and len(elm.get_values_at_anchor(target_anchor)) < MAX_EVALUATIONS:
        
        remaining_time = deadline - time.time() - 0.1 if deadline is not None else np.inf
        if remaining_time < 1:
            logger.info("Timeout observed, stopping outer loop of LCCV")
            break
        
        # initialize stage-specific variables
        eps = max_conf_interval_size_target if t == T else max_conf_interval_size_default
        s_t = schedule[t]
        num_evaluations_at_t = len(elm.get_values_at_anchor(s_t))
        logger.info(f"Running iteration for t = {t}. Anchor point s_t is {s_t}. Remaining time: {remaining_time}s")
        
        ## INNER LOOP: acquire observations at anchor until stability is reached, or just a single one to repair convexity
        while repair_convexity or num_evaluations_at_t < min_evals_for_stability or (elm.get_conf_interval_size_at_target(s_t) > eps and num_evaluations_at_t < MAX_EVALUATIONS):
            
            remaining_time = deadline - time.time() - 0.1 if deadline is not None else np.inf
            if remaining_time < 1:
                logger.info("Timeout observed, stopping inner loop of LCCV")
                break
            
            # unset flag for convexity repair
            repair_convexity = False
            
            # compute next sample
            try:
                seed_used = 13 * (1 + seed) + num_evaluations_at_t
                logger.debug(f"Adding point at size {s_t} with seed is {seed_used}. Remaining time: {remaining_time}s")
                score_train, score_test = elm.compute_and_add_sample(s_t, seed_used, (deadline - time.time() - 0.1) * 1000 if deadline is not None else None, verbose=verbose)
                num_evaluations_at_t += 1
                logger.debug(f"Sample computed successfully. Observed performance was {np.round(score_train, 4)} (train) and {np.round(score_test, 4)} (test).")
            except func_timeout.FunctionTimedOut:
                timeouted = True
                logger.info("Observed timeout. Stopping LCCV.")
                break
            except Exception as e:
                logger.info(f"Observed an exception at anchor {s_t}.\nRaising it to the outside and ignoring this candidate.\nThis is not necessarily a good strategy; depending on the exception, one should try the candidate again on the same or bigger data size, because this can be related to a too small sample size.\nThe exception was: {e}.")
                if exceptions == "raise":
                    raise
                score_train, score_test = np.nan, np.nan
                num_evaluations_at_t += 1
            
            # check wheter a repair is needed
            if num_evaluations_at_t >= min_evals_for_stability and t < T and t > 2:                    
                slopes = elm.get_slope_ranges()
                if len(slopes) < 2:
                    raise Exception(f"There should be two slope ranges for t > 2 (t is {t}), but we observed only 1.")
                if slopes[t - 2] > slopes[t - 1] and len(elm.get_values_at_anchor(schedule[t - 1])) < MAX_EVALUATIONS:
                    repair_convexity = True
                    break

        # check training curve
        if use_train_curve != False:
            
            check_training_curve = (type(use_train_curve) == bool) or (callable(use_train_curve) and use_train_curve(learner_inst, s_t))
            
            if check_training_curve and elm.get_best_worst_train_score() < r:
                logger.info(f"Train curve has value {elm.get_best_worst_train_score()} that is already worse than r = {r}. Stopping.")
                break
        
        # after the last stage, we dont need any more tests
        if t == T:
            logger.info("Last iteration has been finished. Not testing anything else anymore.")
            break
        
        # now decide how to proceed
        if repair_convexity:
            t -= 1
            logger.debug(f"Convexity needs to be repaired, stepping back. t is now {t}")
        elif t >= 2 and elm.get_performance_interval_at_target(target_anchor)[1] < r:
            
            if visualize_lcs:
                logger.debug(f"Visualizing curve")
                elm.visualize(schedule[-1], r)
            
            estimate_for_target_performance = elm.get_performance_interval_at_target(target_anchor)
            optimistic_estimate_for_target_performance = estimate_for_target_performance[1]
            
            # prepare data for cut-off summary
            pessimistic_slope, optimistic_slope = elm.get_slope_range_in_last_segment()
            estimates = elm.get_normal_estimates()
            sizes = sorted(np.unique(elm.df["trainsize"]))
            i = -1
            if min_evals_for_stability > 1:
                while len(elm.df[elm.df["trainsize"] == sizes[i]]) < 2:
                    i -= 1
            last_size = s_t
            normal_estimates_last = estimates[last_size]
            last_conf = normal_estimates_last["conf"]
            
            # inform about cut-off
            logger.info(f"Impossibly reachable. Best possible score by bound is {optimistic_estimate_for_target_performance}. Stopping after anchor s_t = {s_t} and returning nan.")
            logger.debug(f"""Details about stop:
            Data:
            {elm.df}
            Normal Estimates: """ + ''.join(["\n\t\t" + str(s_t) + ": " + (str(estimates[s_t]) if s_t in estimates else "n/a") for s_t in schedule]) + "\n\tSlope Ranges:" + ''.join(["\n\t\t" + str(schedule[i]) + " - " + str(schedule[i + 1]) + ": " +  str(e) for i, e in enumerate(elm.get_slope_ranges())]) + f"""
            Last size: {last_size}
            Optimistic offset at last evaluated anchor {last_size}: {last_conf[1]}
            Optimistic slope from last segment: {optimistic_slope}
            Remaining steps: {(target_anchor - last_size)}
            Estimated interval at target size {target_anchor} (pessimistic, optimistic): {estimate_for_target_performance}""")
            return np.nan, normal_estimates_last["mean"], estimates, elm

        elif not enforce_all_anchor_evaluations and (elm.get_mean_performance_at_anchor(s_t) > r or (t >= 3 and elm.get_lc_estimate_at_target(target_anchor) >= r - MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION)):
            t = T
            if (elm.get_mean_performance_at_anchor(s_t) > r):
                logger.info(f"Current mean is {elm.get_mean_performance_at_anchor(s_t)}, which is already an improvement over r = {r}. Hence, stepping to full size.")
            else:
                logger.info(f"Candidate appears to be competitive (predicted performance at {target_anchor} is {elm.get_lc_estimate_at_target(target_anchor)}. Jumping to last anchor in schedule: {t}")
        else:
            t += 1
            logger.info(f"Finished schedule on {s_t}, and t is now {t}. Performance: {elm.get_normal_estimates(s_t, 4)}.")
            if t < T:
                estimates = elm.get_normal_estimates()
                logger.debug("LC: " + ''.join(["\n\t" + str(s_t) + ": " + (str(estimates[s_t]) if s_t in estimates else "n/a") + ". Avg. runtime: " + str(np.round(np.mean(elm.get_runtimes_at_anchor(s_t) / 1000), 1)) for s_t in schedule if len(elm.get_runtimes_at_anchor(s_t)) > 0]))
                if t > 2:
                    logger.debug(f"Estimate for target size {target_anchor}: {elm.get_performance_interval_at_target(target_anchor)[1]}")
    
    # output final reports
    toc = time.time()
    estimates = elm.get_normal_estimates()
    logger.info(f"Learning Curve Construction Completed. Summary:\n\tRuntime: {int(1000*(toc-tic))}ms.\n\tLC: " + ''.join(["\n\t\t" + str(s_t) + ":\t" + (", ".join([str(k) + ": " + str(np.round(v, 4)) for k, v in estimates[s_t].items()]) if s_t in estimates else "n/a") + ". Avg. runtime: " + str(np.round(np.mean(elm.get_runtimes_at_anchor(s_t) / 1000), 1)) for s_t in schedule if len(elm.get_runtimes_at_anchor(s_t)) > 0]))
    
    # return result depending on observations and configuration
    if len(estimates) == 0 or elm.get_best_worst_train_score() < r:
        logger.info(f"Observed no result or a train performance that is worse than r. In either case, returning nan.")
        return np.nan, np.nan, dict() if len(estimates) == 0 else estimates, elm
    elif len(estimates) < 3:
        max_anchor = max([int(k) for k in estimates])
        if visualize_lcs:
            logger.debug(f"Visualizing curve")
            elm.visualize(schedule[-1], r)
        return estimates[max_anchor]["mean"], estimates[max_anchor]["mean"], estimates, elm
    else:
        max_anchor = max([int(k) for k in estimates])
        target_performance = estimates[max_anchor]["mean"] if t == T or not return_estimate_on_incomplete_runs else elm.get_lc_estimate_at_target(target_anchor)
        logger.info(f"Target performance: {target_performance}")
        if visualize_lcs:
            logger.debug(f"Visualizing curve")
            elm.visualize(schedule[-1], r)
        return target_performance, estimates[max_anchor]["mean"], estimates, elm
