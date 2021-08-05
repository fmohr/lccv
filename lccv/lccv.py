import numpy as np
import pandas as pd
import scipy.stats
import time
import random
import sklearn.metrics
from evalutils import evaluate
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt


def getSlopes(anchor_points, observations):
    slopes = []
    for i, o in enumerate(observations):
        if i > 0:
            slope = (mean(o) - mean(observations[i-1])) / (anchor_points[i] - anchor_points[i-1])
            slopes.append(slope)
    return slopes


def mean(A):
    if len(A) == 0:
        raise Exception("Cannot compute mean for empty set.")
    #return scipy.stats.trim_mean(A, 0.1)
    return np.mean(A)


def getLCApproximation(sizes, scores):
    def ipl(beta):
        a, b, c = tuple(beta.astype(float))
        pl = lambda x: a + b * x **(-c)
        penalty = []
        for i, size in enumerate(sizes):
            penalty.append((pl(size) - scores[i])**2)
        return np.array(penalty)

    a, b, c = tuple(scipy.optimize.least_squares(ipl, np.array([1,1,1]), method="lm").x)
    return lambda x: a + b * x **(-c)


def getStagesAndBudgets(n, k = 10, alpha = .5, gamma = 2, min_anchor_points = 5):
        
    # derive basic sizes
    d = int(np.floor(np.log(.9 * n) / np.log(2)))
    ac = alpha * k
    
    # optimize for c and beta
    c = min_anchor_points + 1
    beta = (alpha / gamma)**(1/(c-min_anchor_points))
    while np.sum([1/(2*beta)**i for i in range(c - min_anchor_points + 1)]) <= 2**(d-c)/alpha:
        c += 1
        beta = (alpha / gamma)**(1/(c-min_anchor_points))
    c -= 1
    beta = (alpha / gamma)**(1/(c-min_anchor_points))
    
    # define anchor points and time budgets
    points = 2**np.array(range(min_anchor_points, c + 1))
    budgets = []
    for i, p in enumerate(points):
        budgets.append(int(np.round(ac / (beta**(c-i - min_anchor_points)))))
    return c, budgets


def get_bootstrap_samples(observations, n, stats=lambda x: np.mean(x)):
    if len(observations) <= 2:
        raise Exception("Cannot compute bootstrap sample of less than 2 observations!")
    bootstraps = []
    observations_as_list = list(observations)
    bootstrap_size = int(0.5 * len(observations_as_list))
    for i in range(n):
        sub_sample = random.sample(observations_as_list, bootstrap_size)
        bootstraps.append(stats(sub_sample))
    return bootstraps


class EmpiricalLearningModel:
    
    def __init__(self, learner, X, y):
        self.learner = learner
        self.X = X
        self.y = y
        self.df = pd.DataFrame([], columns=["trainsize", "seed", "error_rate", "runtime"])
    
    def compute_and_add_sample(self, size, seed = None, timeout = None, verbose = False):
        tic = time.time()
        if seed is None:
            seed = int(tic)
        error_rate = evaluate(sklearn.base.clone(self.learner), self.X, self.y, size, seed, timeout / 1000 if timeout is not None else None, verbose=verbose)
        toc = time.time()
        self.df.loc[len(self.df)] = [size, seed, error_rate, int(np.round(1000 * (toc-tic)))]
        self.df = self.df.astype({"trainsize": int, "seed": int, "runtime": int})
    
    def get_values_at_anchor(self, anchor):
        return self.df[self.df["trainsize"] == anchor]["error_rate"].values
        
    def get_ipl_estimate_at_target(self, target):
        return self.get_ipl()(target)
    
    def get_confidence_interval(self, size):
        dfProbesAtSize = self.df[self.df["trainsize"] == size]
        return 
    
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
            "conf": scipy.stats.norm.interval(0.95, loc=mu, scale=sigma/np.sqrt(len(dfProbesAtSize)))
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

    def plot_model(self, ax = None):
        estimates = self.get_normal_estimates()
        sizes = [s for s in estimates]
        means = [estimates[s]["mean"] for s in sizes]
        lower = [estimates[s]["conf"][0] for s in sizes]
        upper = [estimates[s]["conf"][1] for s in sizes]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(sizes, means)
        ax.fill_between(sizes, lower, upper, alpha=0.2)
    

def lccv(learner_inst, X, y, r = 1.0, timeout=None, base = 2, min_exp = 6, MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION = 0.03, MAX_EVALUATIONS = 10, target = None, return_estimate_on_incomplete_runs=False, max_conf_interval_size_default = 0.1, max_conf_interval_size_target = 0.001, enforce_all_anchor_evaluations=False, seed=0, verbose=True):
    
    if enforce_all_anchor_evaluations:
        if verbose:
            print("All anchor evaluations enforced, setting r to 1.0")
        r = 1.0
    
    # intialize
    tic = time.time()
    deadline = tic + timeout if timeout is not None else None
    elm = EmpiricalLearningModel(learner_inst, X, y)
    
    # configure the exponents and status variables
    if target is None:
        target = int(np.floor(X.shape[0] * 0.9))
    max_exp = np.log(target) / np.log(base)    
    reachable = True
    estimate_history = []
    stable_anchors = []
    
    if verbose:
        print("Running LCCV on " + str(X.shape) + "-shaped data for learner " + str(learner_inst) + " with r = " + str(r) + ". Overview:\n\tmin_exp: " + str(min_exp) + "\n\tmax_exp: " + str(max_exp) + ". Seed is " + str(seed))
    
    # while we can still reach the target value r but have not yet reached it, keep running.
    cur_exp = min_exp
    eval_counter = {}
    timeouted = False
    while reachable and cur_exp <= max_exp and not timeouted and (not max_exp in eval_counter or eval_counter[max_exp] < MAX_EVALUATIONS):
        
        target_estimates = elm.get_normal_estimates(target)
        if np.isnan(target_estimates["conf"][0]) and target_estimates["std"] == 0 and target_estimates["n"] > 2:
            if verbose:
                print("convered, stopping")
            break
        if target_estimates["conf"][1] - target_estimates["conf"][0] < max_conf_interval_size_target:
            if verbose:
                print("convered, stopping")
            break
        
        if verbose:
            print("Next iteration in validation process. cur_exp = " + str(cur_exp) + "/" + str(max_exp) +  " (max_exp). Stable anchors: " + str(stable_anchors))
    
        if cur_exp == max_exp:
            if verbose:
                print("Reached full dataset size. Disabling max number of evaluations (setting it to 10).")
            MAX_EVALUATIONS = 10
        
        # get samples until the variance of the sample mean is believed to be small
        stable = cur_exp in stable_anchors
        size = int(np.round(base ** cur_exp))
        
        while reachable and (not stable or (enforce_all_anchor_evaluations and eval_counter[cur_exp] < MAX_EVALUATIONS)):
            if not cur_exp in eval_counter:
                eval_counter[cur_exp] = 0
            if eval_counter[cur_exp] >= MAX_EVALUATIONS:
                stable = True
                if verbose:
                    print("Maximum number of evaluations reached.")
                break

            try:
                seed_local = eval_counter[cur_exp] if cur_exp in eval_counter else 0
                seed_used = 13 * seed + seed_local
                if verbose:
                    print("Adding point at size " + str(size) + ". Seed is " + str(seed_used) + ". Counter is " + str(eval_counter[cur_exp]))
                elm.compute_and_add_sample(size, seed_used, (deadline - time.time()) * 1000 if deadline is not None else None, verbose=verbose)
                if verbose:
                    print("Sample computed successfully.")
            except FunctionTimedOut:
                timeouted = True
                if verbose:
                    print("Timeouted")
                break

            eval_counter[cur_exp] += 1
            values_at_anchor = elm.get_values_at_anchor(size)
            std = np.std(values_at_anchor)
            if verbose:
                print("std of " + str(values_at_anchor) + ": " + str(std))
            if std == 0:
                stable = len(values_at_anchor) > 2
            else:
                conf_interval = scipy.stats.norm.interval(0.9, loc=np.mean(values_at_anchor), scale=np.std(values_at_anchor)/np.sqrt(len(values_at_anchor)))
                conf_interval_size = (conf_interval[1] - conf_interval[0])
                cond_conf_interval_size = conf_interval_size < max_conf_interval_size_default if cur_exp < max_exp else conf_interval_size < max_conf_interval_size_target
                if verbose:
                    print("Confidence bounds for performance interval at " + str(size) + ": " + str(conf_interval) + ". Size: " + str(conf_interval_size))

                stable = cond_conf_interval_size and (cur_exp - min_exp >= 3 or eval_counter[cur_exp] >= 3)

                if cur_exp == max_exp:
                    reachable = conf_interval[0] <= r
                    if not reachable:
                        if verbose:
                            print("GOAL NOT REACHABLE ANYMORE!")
                        break

            if stable:
                stable_anchors.append(cur_exp)

        if cur_exp < max_exp and len(elm.get_values_at_anchor(size)) >= 1:
            almost_reached = np.mean(elm.get_values_at_anchor(size)) <= r + MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION
            if almost_reached and not enforce_all_anchor_evaluations:
                if cur_exp == max_exp and stable:
                    if verbose:
                        print("Stopping LCCV construction since last stage is stable.")
                    break
                else:
                    if verbose:
                        print("Reached r-score up to a precision of " + str(MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION) + "!")
                        print("Stopping LC construction and setting exponent to maximum " + str(max_exp) + ".")
                    if deadline is None:
                        feasible_target = target
                        if verbose:
                            print("Setting exponent to maximum (no timeout given)")
                    else:
                        remaining_time = deadline - time.time()
                        max_size_in_timeout = elm.get_max_size_for_runtime(remaining_time * 1000)
                        feasible_target = min(target, max_size_in_timeout)
                        if verbose:
                            print("Setting exponent to maximally possible in remaining time " + str(remaining_time) + "s according to current belief.")
                            print("Max size in timeout:",max_size_in_timeout)
                            print("Expected runtime at that size:", elm.predict_runtime(max_size_in_timeout))
                    cur_exp = np.log(feasible_target) / np.log(base)
                    if verbose:
                        print("Feasible target:",feasible_target)
                        print("Setting exponent to", cur_exp)
                    continue
            
        # check whether the goal is still reachable
        if cur_exp > min_exp and not enforce_all_anchor_evaluations:
            
            if cur_exp == max_exp and stable:
                if verbose:
                    print("Stopping LCCV construction since last stage is stable.")
                break
            
            
            bounds = elm.get_performance_interval_at_target(target)
            if verbose:
                print("Estimated bounds for performance interval at " + str(target) + ": " + str(bounds))
            reachable = bounds[1] <= r
            if not reachable:
                pessimistic_slope, optimistic_slope = elm.get_slope_range_in_last_segment()
                if verbose:
                    print("Impossibly reachable, stopping.")
                    print("Details about stop:")
                    print("Data: " + str(elm.df))
                    print("Normal Estimates:" + str(elm.get_normal_estimates()))
                    print("Slope Ranges:", elm.get_slope_ranges())
                    print("Optimistic slope:", optimistic_slope)
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
                if verbose:
                    print("Last size:", last_size)
                    print("Remaining steps:", (target - last_size))
                    print("offset:", last_conf_lower)
                    print("Prediction:", optimistic_slope * (target - last_size) + last_conf_lower)
                
                return np.nan, normal_estimates_last["mean"], normal_estimates, elm
            
            if cur_exp > min_exp + 1 and cur_exp < max_exp:
                estimation = elm.get_ipl_estimate_at_target(target)
                estimate_history.append(estimation)
                if verbose:
                    print("IPL Estimate for target is " + str(estimation) + ".")
                if estimation <= r + MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION:
                    remaining_time = deadline - time.time()
                    max_size_in_timeout = elm.get_max_size_for_runtime(remaining_time * 1000)
                    feasible_target = min(target, max_size_in_timeout)
                    cur_exp = np.log(feasible_target) / np.log(base)
                    MAX_EVALUATIONS = 10
                    
                    if verbose:
                        print("This IS close enough for full evaluation.")
                        print("Setting exponent to maximally possible in remaining time " + str(remaining_time) + "s according to current belief.")
                        print("Max size in timeout:",max_size_in_timeout)
                        print("Expected runtime at that size:", elm.predict_runtime(max_size_in_timeout))
                        print("Feasible target:",feasible_target)
                        print("Setting exponent to", cur_exp)
                        print("Disabling max number of evaluations (setting it to 10).")
                    continue
                else:
                    if verbose:
                        print("This is NOT close enough for full evaluation. Continuing.")
            else:
                if verbose:
                    print("Not enough anchor points yet to get an IPL estimate.")
    
        # check whether after-evaluations are necessary
        slope_ranges = elm.get_slope_ranges()
        ordered_slopes = np.array(np.argsort([s[1] for s in slope_ranges]))
        mismatches = np.where(ordered_slopes != np.array(range(len(ordered_slopes))))[0]
        if len(mismatches) > 0 and not enforce_all_anchor_evaluations:
            act_exp = cur_exp
            cur_exp = min_exp + min(mismatches)
            if verbose:
                print("Found mismatches in slope ordering:", ordered_slopes, mismatches, "Initializing reset with exp " + str(cur_exp))
                print(cur_exp in stable_anchors)
                print(cur_exp in eval_counter)
            while cur_exp in stable_anchors or (cur_exp in eval_counter and eval_counter[cur_exp] >= MAX_EVALUATIONS):
                if verbose:
                    print("Exp " + str(cur_exp) + " is stable or is in the eval_counter with a high enough value.")
                cur_exp += 1
            if cur_exp < act_exp:
                if verbose:
                    print("Going back from exp " + str(act_exp) + " to " + str(cur_exp) + ". Stable anchors are " + str(stable_anchors) + ". Eval counter is: " + str(eval_counter) + ". MAX_EVALUATIONS is: " + str(MAX_EVALUATIONS))
            elif cur_exp > max_exp:
                cur_exp = max_exp
                if verbose:
                    print("Stepping to maximum exponent " + str(cur_exp))
            else:
                if verbose:
                    print("Stepping to exponent " + str(cur_exp))
        else:
            cur_exp += 1
            if cur_exp > max_exp:
                cur_exp = max_exp
    
    toc = time.time()
    if verbose:
        print("Estimation process finished, preparing result.")
    
    estimates = elm.get_normal_estimates()
    if verbose:
        print("Learning Curve Construction Completed. Conditions:\n\tReachable: " + str(reachable) + "\n\tTimeout: " + str(timeouted))
        print("Estimate History:", estimate_history)
        print("LC:", estimates)
        print("Runtime:", toc-tic, "Expected runtime on", target,":",elm.predict_runtime(target))
    if len(estimates) == 0:
        return np.nan, np.nan, [], elm
    elif len(estimates) < 3:
        max_anchor = max([int(k) for k in estimates])
        return estimates[max_anchor]["mean"], estimates[max_anchor]["mean"], estimates, elm
    else:
        max_anchor = max([int(k) for k in estimates])
        target_performance = estimates[max_anchor]["mean"] if cur_exp == max_exp or not return_estimate_on_incomplete_runs else elm.get_ipl_estimate_at_target(target)
        if verbose:
            print("Target performance:", target_performance)
        return target_performance, estimates[max_anchor]["mean"], estimates, elm
