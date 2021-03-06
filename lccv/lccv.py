import typing
import logging

import numpy as np
import pandas as pd
import scipy.stats
import time
import sklearn.metrics
import func_timeout

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
    
    def __init__(self, learner, X, y, n_test, seed, fix_train_test_folds):
        
        # set up logger
        self.logger = logging.getLogger('elm')
        
        self.learner = learner
        self.active_seed = seed
        self.fix_train_test_folds = fix_train_test_folds
        
        if fix_train_test_folds:
            self.X_train, self.y_train, self.X_test, self.y_test = _partition_train_test_data(X, y, n_test, seed)
            self.logger.info(f"Train labels: \n{self.y_train}")
            self.logger.info(f"Test labels: \n{self.y_test}")
        else:
            self.X = X
            self.y = y
            self.n_test = n_test
        self.df = pd.DataFrame([], columns=["trainsize", "seed", "error_rate", "runtime"])
        
        

    def evaluate(self, learner_inst, size, timeout, verbose):

        self.active_seed += 1
        self.logger.debug("Computing trainning data")
        
        # obtain train and test data (depending on configuration)
        if self.fix_train_test_folds:
            self.logger.info("Re-using pre-defined train and test folds")
            X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test
        else:
            self.logger.info("Dynamically creating a train and test fold")
            X_train, y_train, X_test, y_test = _partition_train_test_data(self.X, self.y, self.n_test, self.active_seed)
        
        indices = np.random.choice(X_train.shape[0], size, replace=False)
        X_train = X_train[indices].copy() # these copy actions could maybe be inefficient but are currently required to be stable w.r.t. the copy=false option for some pre-processors
        y_train = y_train[indices].copy()
        
        self.logger.info(f"Training {format_learner(learner_inst)} on data of shape {X_train.shape}. Timeout is {timeout}")
        start = time.time()
        if timeout is None:
            learner_inst.fit(X_train, y_train)
        else:
            func_timeout.func_timeout(timeout, learner_inst.fit, (X_train, y_train))
        end = time.time()
        self.logger.debug(f"Training ready after {int((end - start) * 1000)}ms. Now obtaining predictions.")
        y_hat = learner_inst.predict(X_test.copy())
        error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
        end = time.time()
        self.logger.info(f"Evaluation ready after {int((end - start) * 1000)}ms. Error rate of model on {y_hat.shape[0]} validation/test instances is {error_rate}.")
        return error_rate
    
    def compute_and_add_sample(
            self, size, seed=None, timeout=None, verbose=False):
        tic = time.time()
        # TODO: important to check whether this is always a different order
        error_rate = self.evaluate(
            sklearn.base.clone(self.learner), size,
            timeout / 1000 if timeout is not None else None, verbose)
        toc = time.time()
        runtime = int(np.round(1000 * (toc-tic)))
        self.logger.debug(f"Sample value computed within {runtime}ms")
        self.df.loc[len(self.df)] = [size, seed, error_rate, runtime]
        self.df = self.df.astype({"trainsize": int, "seed": int, "runtime": int})
    
    def get_values_at_anchor(self, anchor):
        return self.df[self.df["trainsize"] == anchor]["error_rate"].values
    
    def get_mean_performance_at_anchor(self, anchor):
        return np.mean(self.get_values_at_anchor(anchor))
    
    def get_conf_interval_size_at_target(self, target):
        if len (self.df[self.df["trainsize"] == target]) == 0:
            return 1
        ci = self.get_normal_estimates(size = target)["conf"]
        return ci[1] - ci[0]
    
    def get_ipl_estimate_at_target(self, target):
        return self.get_ipl()(target)
    
    def get_normal_estimates(self, size = None, round_precision=100):
        
        if size is None:
            sizes = sorted(np.unique(self.df["trainsize"]))
            out = {}
            for size in sizes:
                out[int(size)] = self.get_normal_estimates(size)
            return out
    
        dfProbesAtSize = self.df[self.df["trainsize"] == size]
        mu = np.mean(dfProbesAtSize["error_rate"])
        sigma = np.std(dfProbesAtSize["error_rate"])
        return {
            "n": len(dfProbesAtSize["error_rate"]),
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
                s1 = sizes[i - 1]
                s2 = sizes[i]
                
                if est[s1]["n"] > 1:
                    lower_prev_last = est[s1]["conf"][0]
                    upper_prev_last = est[s1]["conf"][1]
                else:
                    lower_prev_last = upper_prev_last = est[s1]["mean"]
                if est[s2]["n"] > 1:
                    lower_last = est[s2]["conf"][0]
                    upper_last = est[s2]["conf"][1]
                else:
                    lower_last = upper_last = est[s2]["mean"]
                ranges.append((min(0, (upper_last - lower_prev_last) / (s2 - s1)), min(0, (lower_last - upper_prev_last) / (s2 - s1))))
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
        return pessimistic_slope * (target - last_size) + last_conf_upper, optimistic_slope * (target - last_size) + last_conf_lower
        
    def get_ipl(self):
        sizes = sorted(list(pd.unique(self.df["trainsize"])))
        scores = [np.mean(self.df[self.df["trainsize"] == s]["error_rate"]) for s in sizes]
        def ipl(beta):
            a, b, c = tuple(beta.astype(float))
            pl = lambda x: a + b * x **(-c)
            penalty = []
            for i, size in enumerate(sizes):
                penalty.append((pl(size) - scores[i])**2)
            return np.array(penalty)

        a, b, c = tuple(scipy.optimize.least_squares(ipl, np.array([1,1,1]), method="lm").x)
        return lambda x: a + b * x **(-c)
    
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
    

def lccv(learner_inst, X, y, r=1.0, timeout=None, base=2, min_exp=6, MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION=0.03, MAX_EVALUATIONS=10, target_anchor=.9, return_estimate_on_incomplete_runs=False, max_conf_interval_size_default=0.1, max_conf_interval_size_target=0.001, enforce_all_anchor_evaluations=False, seed=0, verbose=False, logger=None, min_evals_for_stability=3,fix_train_test_folds=False):
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
    
    if X.shape[0] <= 0:
        raise Exception(f"Recieved dataset with non-positive number of instances. Shape is {X.shape}")

    # configure the exponents and status variables    print(target_anchor)
    if target_anchor < 1:
        target_anchor = int(np.floor(X.shape[0] * target_anchor))
    
    # initialize important variables and datastructures
    max_exp = np.log(target_anchor) / np.log(base)
    schedule = [base**i for i in list(range(min_exp, int(np.ceil(max_exp))))] + [target_anchor]
    slopes = (len(schedule) - 1) * [np.nan]
    elm = EmpiricalLearningModel(learner_inst, X, y, X.shape[0] - target_anchor, seed, fix_train_test_folds)
    T = len(schedule) - 1
    t = 0 if r < 1 or enforce_all_anchor_evaluations else T
    repair_convexity = False
    
    # announce start event together with state variable values
    logger.info(f"""Running LCCV on {X.shape}-shaped data. Overview:
    learner: {format_learner(learner_inst)}
    r: {r}
    min_exp: {min_exp}
    max_exp: {max_exp}
    Seed is {seed}
    t_0: {t}
    Schedule: {schedule}""")
    
    ## MAIN LOOP
    while t <= T and elm.get_conf_interval_size_at_target(target_anchor) > max_conf_interval_size_target and len(elm.get_values_at_anchor(target_anchor)) < MAX_EVALUATIONS:
        
        remaining_time = deadline - time.time() if deadline is not None else np.inf
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
            
            remaining_time = deadline - time.time() if deadline is not None else np.inf
            if remaining_time < 1:
                logger.info("Timeout observed, stopping inner loop of LCCV")
                break
            
            # unset flag for convexity repair
            repair_convexity = False
            
            # compute next sample
            try:
                seed_used = 13 * (1 + seed) + num_evaluations_at_t
                logger.debug(f"Adding point at size {s_t} with seed is {seed_used}. Remaining time: {remaining_time}s")
                elm.compute_and_add_sample(s_t, seed_used, (deadline - time.time()) * 1000 if deadline is not None else None, verbose=verbose)
                num_evaluations_at_t += 1
                logger.debug("Sample computed successfully.")
            except func_timeout.FunctionTimedOut:
                timeouted = True
                logger.info("Observed timeout. Stopping LCCV.")
                break
            
            # check wheter a repair is needed
            if num_evaluations_at_t >= min_evals_for_stability and t < T:
                if t > 2:
                    slopes = elm.get_slope_ranges()
                    if len(slopes) < 2:
                        raise Exception(f"There should be two slope ranges for t > 2 (t is {t}), but we observed only 1.")
                    if slopes[t - 2] < slopes[t - 1] and len(elm.get_values_at_anchor(schedule[t - 1])) < MAX_EVALUATIONS:
                        repair_convexity = True
                        break
        
        # after the last stage, we dont need any more tests
        if t == T:
            logger.info("Last iteration has been finished. Not testing anything else anymore.")
            break
        
        # now decide how to proceed
        if repair_convexity:
            t -= 1
            logger.debug(f"Convexity needs to be repaired, stepping back. t is now {t}")
        elif t >= 2 and elm.get_performance_interval_at_target(target_anchor)[1] >= r:
            optimistic_estimate_for_target_performance = elm.get_performance_interval_at_target(target_anchor)[1]
            
            # prepare data for cut-off summary
            pessimistic_slope, optimistic_slope = elm.get_slope_range_in_last_segment()
            estimates = elm.get_normal_estimates()
            sizes = sorted(np.unique(elm.df["trainsize"]))
            i = -1
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
            Optimistic offset at last evaluated anchor {last_size}: {last_conf[0]}
            Optimistic slope from last segment: {optimistic_slope}
            Remaining steps: {(target_anchor - last_size)}
            Most optimistic value possible at target size {target_anchor}: {optimistic_slope * (target_anchor - last_size) + last_conf[0]}""")
            return np.nan, normal_estimates_last["mean"], estimates, elm

        elif not enforce_all_anchor_evaluations and (elm.get_mean_performance_at_anchor(s_t) < r or (t >= 3 and elm.get_ipl_estimate_at_target(target_anchor) <= r + MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION)):
            t = T
            if (elm.get_mean_performance_at_anchor(s_t) < r):
                logger.info(f"Current mean is {elm.get_mean_performance_at_anchor(s_t)}, which is already an improvement over r = {r}. Hence, stepping to full size.")
            else:
                logger.info(f"Candidate appears to be competitive (predicted performance at {target_anchor} is {elm.get_ipl_estimate_at_target(target_anchor)}. Jumping to last anchor in schedule: {t}")
        else:
            t += 1
            logger.info(f"Finished schedule on {s_t}, and t is now {t}. Performance: {elm.get_normal_estimates(s_t, 4)}.")
            if t < T:
                estimates = elm.get_normal_estimates()
                logger.debug("LC: " + ''.join(["\n\t" + str(s_t) + ": " + (str(estimates[s_t]) if s_t in estimates else "n/a") for s_t in schedule]))
                if t > 2:
                    logger.debug(f"Estimate for target size {target_anchor}: {elm.get_performance_interval_at_target(target_anchor)[1]}")
    
    # output final reports
    toc = time.time()
    estimates = elm.get_normal_estimates()
    logger.info(f"Learning Curve Construction Completed. Summary:\n\tRuntime: {int(1000*(toc-tic))}ms.\n\tLC: " + ''.join(["\n\t\t" + str(s_t) + ": " + (str(estimates[s_t]) if s_t in estimates else "n/a") for s_t in schedule]))
    
    # return result depending on observations and configuration
    if len(estimates) == 0:
        return np.nan, np.nan, dict(), elm
    elif len(estimates) < 3:
        max_anchor = max([int(k) for k in estimates])
        return estimates[max_anchor]["mean"], estimates[max_anchor]["mean"], estimates, elm
    else:
        max_anchor = max([int(k) for k in estimates])
        target_performance = estimates[max_anchor]["mean"] if t == T or not return_estimate_on_incomplete_runs else elm.get_ipl_estimate_at_target(target_anchor)
        logger.info(f"Target performance: {target_performance}")
        return target_performance, estimates[max_anchor]["mean"], estimates, elm
