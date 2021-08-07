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
    indices = np.arange(len(features))
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
        indices = np.random.choice(len(self.X_train), size, replace=False)

        self.logger.info("Training " + str(learner_inst) + " on data of shape " + str(self.X_train.shape))
        if deadline is None:
            learner_inst.fit(self.X_train[indices], self.y_train[indices])
        else:
            func_timeout.func_timeout(deadline - time.time(), learner_inst.fit,
                                      (self.X_train[indices], self.y_train[indices]))

        y_hat = learner_inst.predict(self.X_test)
        error_rate = 1 - sklearn.metrics.accuracy_score(self.y_test, y_hat)
        self.logger.info("Training ready. Obtaining predictions for " + str(self.X_test.shape[0]) + " instances. Error rate of model on " + str(len(y_hat)) + " instances is " + str(error_rate))
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
        
    def get_ipl_estimate_at_target(self, target):
        return self.get_ipl()(target)
    
    def get_normal_estimates(self, size = None):
        
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
            "mean": mu,
            "std": sigma,
            "conf": scipy.stats.norm.interval(0.95, loc=mu, scale=sigma/np.sqrt(len(dfProbesAtSize))) if sigma > 0 else (mu, mu)
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
    
    if enforce_all_anchor_evaluations:
        logger.info("All anchor evaluations enforced, setting r to 1.0")
        r = 1.0
    
    # intialize
    tic = time.time()
    deadline = tic + timeout if timeout is not None else None

    # configure the exponents and status variables
    if target_anchor is None:
        target_anchor = int(np.floor(X.shape[0] * 0.9))

    elm = EmpiricalLearningModel(learner_inst, X, y, X.shape[0] - target_anchor, seed)

    max_exp = np.log(target_anchor) / np.log(base)    
    reachable = True
    estimate_history = []
    stable_anchors = []
    
    logger.info(f"""Running LCCV on {X.shape}-shaped data for learner {learner_inst} with r = {r}. Overview:
    \n\tmin_exp: {min_exp}
    \n\tmax_exp: {max_exp}
    \n\tSeed is {seed}""")
    
    # while we can still reach the target value r but have not yet reached it, keep running.
    cur_exp = min_exp
    eval_counter = {}
    timeouted = False
    while reachable and cur_exp <= max_exp and not timeouted and (not max_exp in eval_counter or eval_counter[max_exp] < MAX_EVALUATIONS):
        
        target_estimates = elm.get_normal_estimates(target_anchor)
        if np.isnan(target_estimates["conf"][0]) and target_estimates["std"] == 0 and target_estimates["n"] > 2:
            logger.info("convered, stopping")
            break
        if target_estimates["conf"][1] - target_estimates["conf"][0] < max_conf_interval_size_target:
            logger.info("convered, stopping")
            break
        
        if cur_exp == max_exp:
            logger.info("Reached full dataset size. Disabling max number of evaluations (setting it to 10).")
            MAX_EVALUATIONS = 10
        
        # get samples until the variance of the sample mean is believed to be small
        stable = cur_exp in stable_anchors
        size = int(np.round(base ** cur_exp))
        logger.info(f"Entering stage for anchor point size {size}. That is exponent {cur_exp}. Currently considered maximum size is {max_exp}. The stable anchors are {stable_anchors}")
        
        while reachable and (not stable or (enforce_all_anchor_evaluations and eval_counter[cur_exp] < MAX_EVALUATIONS)):
            if not cur_exp in eval_counter:
                eval_counter[cur_exp] = 0
            if eval_counter[cur_exp] >= MAX_EVALUATIONS:
                stable = True
                logger.info("Maximum number of evaluations reached.")
                break

            try:
                seed_local = eval_counter[cur_exp] if cur_exp in eval_counter else 0
                seed_used = 13 * seed + seed_local
                logger.info("Adding point at size " + str(size) + ". Seed is " + str(seed_used) + ". Counter is " + str(eval_counter[cur_exp]))
                elm.compute_and_add_sample(size, seed_used, (deadline - time.time()) * 1000 if deadline is not None else None, verbose=verbose)
                logger.debug("Sample computed successfully.")
            except func_timeout.FunctionTimedOut:
                timeouted = True
                logger.debug("Timeouted")
                break

            eval_counter[cur_exp] += 1
            values_at_anchor = elm.get_values_at_anchor(size)
            std = np.std(values_at_anchor)
            logger.debug("std of " + str(values_at_anchor) + ": " + str(std))
            if std == 0:
                stable = len(values_at_anchor) > 2
            else:
                conf_interval = scipy.stats.norm.interval(0.9, loc=np.mean(values_at_anchor), scale=np.std(values_at_anchor)/np.sqrt(len(values_at_anchor)))
                conf_interval_size = (conf_interval[1] - conf_interval[0])
                cond_conf_interval_size = conf_interval_size < max_conf_interval_size_default if cur_exp < max_exp else conf_interval_size < max_conf_interval_size_target
                logger.debug("Confidence bounds for performance interval at " + str(size) + ": " + str(conf_interval) + ". Size: " + str(conf_interval_size))

                stable = cond_conf_interval_size and (cur_exp - min_exp >= 3 or eval_counter[cur_exp] >= min_evals_for_stability)

                if cur_exp == max_exp:
                    reachable = conf_interval[0] <= r
                    if not reachable:
                        logger.info("GOAL NOT REACHABLE ANYMORE!")
                        break

            if stable:
                stable_anchors.append(cur_exp)

        if cur_exp < max_exp and len(elm.get_values_at_anchor(size)) >= 1:
            mean_score_at_last_anchor = np.mean(elm.get_values_at_anchor(size))
            almost_reached =  mean_score_at_last_anchor <= r + MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION
            if almost_reached and not enforce_all_anchor_evaluations:
                if cur_exp == max_exp and stable:
                    logger.info("Stopping LCCV construction since last stage is stable.")
                    break
                else:
                    if deadline is None:
                        feasible_target = target_anchor
                        logger.debug("Setting exponent to maximum (no timeout given)")
                    else:
                        remaining_time = deadline - time.time()
                        max_size_in_timeout = elm.get_max_size_for_runtime(remaining_time * 1000)
                        feasible_target = min(target_anchor, max_size_in_timeout)
                        logger.debug(f"Setting exponent to maximally possible in remaining time {remaining_time}s according to current belief.")
                        logger.debug(f"Max size in timeout: {max_size_in_timeout}")
                        logger.debug(f"Expected runtime at that size: {elm.predict_runtime(max_size_in_timeout)}")
                    cur_exp = np.log(feasible_target) / np.log(base)
                    logger.debug(f"Feasible target: {feasible_target}")
                    logger.info(f"Reached r-score {r} up to a precision of {MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION} (curent score is {np.round(mean_score_at_last_anchor, 4)})! Stopping LC construction and setting exponent to maximum value possible considering timeouts etc. Exponent is now {cur_exp}.")
                    continue
            
        # check whether the goal is still reachable
        if cur_exp > min_exp and not enforce_all_anchor_evaluations:
            
            if cur_exp == max_exp and stable:
                logger.info(f"Stopping LCCV construction since last stage is stable (with normal estimates {elm.get_normal_estimates(size)}")
                break
            
            
            bounds = elm.get_performance_interval_at_target(target_anchor)
            logger.info("Estimated bounds for performance interval at " + str(target_anchor) + ": " + str(bounds))
            reachable = bounds[1] <= r
            if not reachable:
                pessimistic_slope, optimistic_slope = elm.get_slope_range_in_last_segment()
                logger.info("Impossibly reachable, stopping.")
                logger.debug(f"Details about stop:\nData:\n {elm.df} \n\tNormal Estimates: {elm.get_normal_estimates()}\n\tSlope Ranges: {elm.get_slope_ranges()}\n\tOptimistic slope: {optimistic_slope}")
                sizes = sorted(np.unique(elm.df["trainsize"]))
                i = -1
                while len(elm.df[elm.df["trainsize"] == sizes[i]]) < 2:
                    i -= 1
                last_size = sizes[i]
                normal_estimates = elm.get_normal_estimates()
                normal_estimates_last = normal_estimates[last_size]
                last_conf = normal_estimates_last["conf"]
                last_conf_lower = last_conf[0]
                last_conf_upper = last_conf[1]
                if np.isnan(last_conf_lower):
                    last_conf_lower = last_conf_upper = normal_estimates_last["mean"]
                    last_conf = (last_conf_lower, last_conf_upper)
                logger.debug(f"""Summary of cut-off run:
                            \tLast size: {last_size}
                            \tRemaining steps: {(target_anchor - last_size)}
                            \toffset: {last_conf_lower}
                            \tPrediction: {optimistic_slope * (target_anchor - last_size) + last_conf_lower}""")
                logger.info(f"Returning nan and normal estimates: {normal_estimates_last['mean']}, {normal_estimates}, {elm}")
                return np.nan, normal_estimates_last["mean"], normal_estimates, elm
            
            if cur_exp > min_exp + 1 and cur_exp < max_exp:
                estimation = elm.get_ipl_estimate_at_target(target_anchor)
                estimate_history.append(estimation)
                logger.info(f"IPL Estimate for target is {estimation}.")
                if estimation <= r + MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION:
                    remaining_time = deadline - time.time()
                    max_size_in_timeout = elm.get_max_size_for_runtime(remaining_time * 1000)
                    feasible_target = min(target, max_size_in_timeout)
                    cur_exp = np.log(feasible_target) / np.log(base)
                    MAX_EVALUATIONS = 10
                    
                    logger.info("This IS close enough for full evaluation.")
                    logger.info(f"Setting exponent to maximally possible in remaining time {remaining_time}s according to current belief.")
                    logger.info(f"Max size in timeout: {max_size_in_timeout}")
                    logger.info(f"Expected runtime at that size: {elm.predict_runtime(max_size_in_timeout)}")
                    logger.info(f"Feasible target: {feasible_target}")
                    logger.info(f"Setting exponent to {cur_exp}")
                    logger.info("Disabling max number of evaluations (setting it to 10).")
                    continue
                else:
                    logger.info("This is NOT close enough for full evaluation. Continuing.")
            else:
                logger.info("Not enough anchor points yet to get an IPL estimate.")
    
        # check whether after-evaluations are necessary
        slope_ranges = elm.get_slope_ranges()
        ordered_slopes = np.array(np.argsort([s[1] for s in slope_ranges]))
        mismatches = np.where(ordered_slopes != np.array(range(len(ordered_slopes))))[0]
        if len(mismatches) > 0 and not enforce_all_anchor_evaluations:
            act_exp = cur_exp
            cur_exp = min_exp + min(mismatches)
            logger.info("Found mismatches in slope ordering:", ordered_slopes, mismatches, "Initializing reset with exp " + str(cur_exp))
            logger.info(cur_exp in stable_anchors)
            logger.info(cur_exp in eval_counter)
            while cur_exp in stable_anchors or (cur_exp in eval_counter and eval_counter[cur_exp] >= MAX_EVALUATIONS):
                logger.info("Exp " + str(cur_exp) + " is stable or is in the eval_counter with a high enough value.")
                cur_exp += 1
            if cur_exp < act_exp:
                logger.info("Going back from exp " + str(act_exp) + " to " + str(cur_exp) + ". Stable anchors are " + str(stable_anchors) + ". Eval counter is: " + str(eval_counter) + ". MAX_EVALUATIONS is: " + str(MAX_EVALUATIONS))
            elif cur_exp > max_exp:
                cur_exp = max_exp
                logger.info("Stepping to maximum exponent " + str(cur_exp))
            else:
                logger.info("Stepping to exponent " + str(cur_exp))
        else:
            cur_exp += 1
            if cur_exp > max_exp:
                cur_exp = max_exp
    
    toc = time.time()
    logger.info("Estimation process finished, preparing result.")
    
    estimates = elm.get_normal_estimates()
    logger.info("Learning Curve Construction Completed. Conditions:\n\tReachable: " + str(reachable) + "\n\tTimeout: " + str(timeouted))
    logger.info(f"Estimate History: {estimate_history}")
    logger.info(f"LC: {estimates}")
    logger.info(f"Runtime: {toc-tic}. Expected runtime on {target_anchor}: {elm.predict_runtime(target_anchor)}")
    if len(estimates) == 0:
        return np.nan, np.nan, [], elm
    elif len(estimates) < 3:
        max_anchor = max([int(k) for k in estimates])
        return estimates[max_anchor]["mean"], estimates[max_anchor]["mean"], estimates, elm
    else:
        max_anchor = max([int(k) for k in estimates])
        target_performance = estimates[max_anchor]["mean"] if cur_exp == max_exp or not return_estimate_on_incomplete_runs else elm.get_ipl_estimate_at_target(target_anchor)
        logger.info(f"Target performance: {target_performance}")
        return target_performance, estimates[max_anchor]["mean"], estimates, elm
