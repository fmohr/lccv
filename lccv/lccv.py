import typing
import logging

import numpy as np
import pandas as pd
import scipy.stats
import time
import sklearn.metrics
import func_timeout


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
    
    def __init__(self, learner, X, y, n_test, seed):
        self.learner = learner
        self.X_train, self.y_train, self.X_test, self.y_test = _partition_train_test_data(X, y, n_test, seed)
        self.df = pd.DataFrame([], columns=["trainsize", "seed", "error_rate", "runtime"])
        self.logger = logging.getLogger('elm')
        self.logger.setLevel(logging.DEBUG)

    def evaluate(self, learner_inst, size, timeout, verbose):
        deadline = None if timeout is None else time.time() + timeout
        indices = np.random.choice(self.X_train.shape[0], size, replace=False)

        self.logger.info("Training " + str(learner_inst) + " on data of shape " + str(self.X_train.shape))
        if deadline is None:
            learner_inst.fit(self.X_train[indices], self.y_train[indices])
        else:
            func_timeout.func_timeout(deadline - time.time(), learner_inst.fit,
                                      (self.X_train[indices], self.y_train[indices]))

        y_hat = learner_inst.predict(self.X_test)
        error_rate = 1 - sklearn.metrics.accuracy_score(self.y_test, y_hat)
        self.logger.info("Training ready. Obtaining predictions for " + str(self.X_test.shape[0]) + " instances. Error rate of model on " + str(y_hat.shape[0]) + " instances is " + str(error_rate))
        return error_rate
    
    def compute_and_add_sample(
            self, size, seed=None, timeout=None, verbose=False):
        tic = time.time()
        # TODO: important to check whether this is always a different order
        error_rate = self.evaluate(
            sklearn.base.clone(self.learner), size,
            timeout / 1000 if timeout is not None else None, verbose)
        toc = time.time()
        self.df.loc[len(self.df)] = [
            size, seed, error_rate, int(np.round(1000 * (toc-tic)))]
        self.df = self.df.astype({"trainsize": int, "seed": int, "runtime": int})
    
    def get_values_at_anchor(self, anchor):
        return self.df[self.df["trainsize"] == anchor]["error_rate"].values
    
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
    

def lccv(learner_inst, X, y, r=1.0, timeout=None, base=2, min_exp=6, MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION=0.03, MAX_EVALUATIONS=10, target_anchor=None, return_estimate_on_incomplete_runs=False, max_conf_interval_size_default=0.1, max_conf_interval_size_target=0.001, enforce_all_anchor_evaluations=False, seed=0, verbose=False, logger=None, min_evals_for_stability=5):
    
    # create standard logger if none is given
    if logger is None:
        logger = logging.getLogger('lccv')
    
    # intialize
    tic = time.time()
    deadline = tic + timeout if timeout is not None else None

    # configure the exponents and status variables
    if target_anchor is None:
        target_anchor = int(np.floor(X.shape[0] * 0.9))
    
    
    # initialize important variables and datastructures
    max_exp = np.log(target_anchor) / np.log(base)
    schedule = [base**i for i in list(range(min_exp, int(np.ceil(max_exp))))] + [target_anchor]
    slopes = (len(schedule) - 1) * [np.nan]
    elm = EmpiricalLearningModel(learner_inst, X, y, X.shape[0] - target_anchor, seed)
    T = len(schedule) - 1
    t = 0 if r < 1 or enforce_all_anchor_evaluations else T
    repair_convexity = False
    
    # announce start event together with state variable values
    logger.info(f"""Running LCCV on {X.shape}-shaped data for learner {learner_inst} with r = {r}. Overview:
    min_exp: {min_exp}
    max_exp: {max_exp}
    Seed is {seed}
    t_0: {t}
    Schedule: {schedule}""")
    
    ## MAIN LOOP
    while t <= T and elm.get_conf_interval_size_at_target(target_anchor) > max_conf_interval_size_target and len(elm.get_values_at_anchor(target_anchor)) < MAX_EVALUATIONS:    
        
        # initialize stage-specific variables
        eps = max_conf_interval_size_target if t == T else max_conf_interval_size_default
        s_t = schedule[t]
        num_evaluations_at_t = len(elm.get_values_at_anchor(s_t))
        logger.debug(f"Running iteration for t = {t}. Anchor point s_t is {s_t}")
        
        ## INNER LOOP: acquire observations at anchor until stability is reached, or just a single one to repair convexity
        while repair_convexity or num_evaluations_at_t < min_evals_for_stability or (elm.get_conf_interval_size_at_target(s_t) > eps and num_evaluations_at_t < MAX_EVALUATIONS):
            
            # unset flag for convexity repair
            repair_convexity = False
            
            # compute next sample
            try:
                seed_used = 13 * (1 + seed) + num_evaluations_at_t
                logger.debug(f"Adding point at size {s_t} with seed is {seed_used}.")
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
            logger.info("Impossibly reachable, stopping and returning nan.")
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

        elif not enforce_all_anchor_evaluations and t >= 3 and elm.get_ipl_estimate_at_target(target_anchor) <= r + MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION:
            t = T
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
        return np.nan, np.nan, [], elm
    elif len(estimates) < 3:
        max_anchor = max([int(k) for k in estimates])
        return estimates[max_anchor]["mean"], estimates[max_anchor]["mean"], estimates, elm
    else:
        max_anchor = max([int(k) for k in estimates])
        target_performance = estimates[max_anchor]["mean"] if t == T or not return_estimate_on_incomplete_runs else elm.get_ipl_estimate_at_target(target_anchor)
        logger.info(f"Target performance: {target_performance}")
        return target_performance, estimates[max_anchor]["mean"], estimates, elm
