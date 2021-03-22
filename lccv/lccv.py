import numpy as np
import scipy.stats
import time
import random
from tqdm.notebook import tqdm
import sklearn.metrics
from evalutils import evaluate

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

'''
 The learning-curve based validation mechanism
 
 Params:
     - learner: the learner *class*
     - X, y: the data for validation
     - target_size: The training size relevant for making the model selection decision (default: 90% of the given data)
     - r: reference score that needs to be achieved so that the exact score is required
     - min_stages: minimum number of anchor points that should be used
     - timeout: timeout for the validation process (in seconds)
     - k: number of repeats in the base evaluation that is tried to beat
     - min_exp: minimum exponent (of 2) used to compute the smallest anchor point (default 6, i.e. the smallest anchor is 2^6 = 64)
     - alpha: factor by which the *last* stage should exceed the base repeats k
     - gamma: factor by which the *first* stage should exceed the base repeats k
'''
def lccv(learner, X, y, target_size=None, r = 0.0, min_stages = 3, timeout=60, k = 10, min_exp = 6, alpha = .5, gamma=2):
    
    deadline = time.time() + timeout
    print("Running LCCV with timeout",timeout)
    
    n = X.shape[0]
    if target_size is None:
        target_size = 0.9 * n
        print("Setting target size to " + str(target_size) + " as 90% of the original dataset.")
    
    # compute budgets and phases
    max_exp, repeats = getStagesAndBudgets(X.shape[0], gamma)
    print(min_exp, max_exp, repeats)
    anchors = list(range(min_exp, max_exp + 1))
    
    # start LCCV algorithm
    
    observations = []
    mean_observations = []
    
    for stage_id, exp in enumerate(tqdm(anchors)):
        if time.time() > deadline:
            break
        num_examples = 2**exp
        num_evaluations = repeats[stage_id]
        print ("Running stage for " + str(num_examples) + " examples. At most " + str(num_evaluations) + " evaluations will be allowed.")
        
        stabilized = False
        
        observations_at_anchor = []
        variances = []
        
        base_evaluations = 0
        while not stabilized and base_evaluations < num_evaluations and time.time() < deadline:
            observations_at_anchor.append(evaluate(learner, X, y, num_examples))
            
            # criterion 1: low variance in mean estimation
            if len(observations_at_anchor) > 2:
                bootstrap_samples = get_bootstrap_samples(observations_at_anchor, n = 100)
                variances.append(np.var(bootstrap_samples))
                stabilized = variances[-1] < 0.00001
            base_evaluations += 1
        
        observations.append(observations_at_anchor)
        mean_observations.append(mean(observations_at_anchor))
        print("Obtained low-variance result for stage " + str(exp))
        
        # "repair" curve until convex
        print(len(mean_observations))
        expected = np.array(range(len(mean_observations)))
        mismatches = (-np.array(mean_observations)).argsort() != expected
        reevaluations = 0
        while np.count_nonzero(mismatches) and len(observations[stage_id]) < repeats[stage_id] and time.time() < deadline:
            reevaluate = np.where(mismatches)[0]
            print("Found mismatches.", (-np.array(mean_observations)).argsort(), expected, mismatches, "Re-Evaluating", reevaluate)
            any_adjusted = False
            for i in reevaluate:
                if len(observations[i]) < repeats[i]:
                    print("Re-Evaluting", 2**anchors[i])
                    new_score = evaluate(learner, X, y, 2**anchors[i])
                    observations[i].append(new_score)
                    mean_prev = mean_observations[i]
                    mean_observations[i] = mean(observations[i])
                    print("Updated mean", mean_observations[i], "previously was",mean_prev)
                    any_adjusted = True
            if not any_adjusted:
                break
            sorted_anchor_indices = (-np.array(mean_observations)).argsort()
            print(sorted_anchor_indices, mean_observations)
            mismatches = sorted_anchor_indices != expected
            reevaluations += 1
        
        
        slopes = getSlopes([2**e for e in anchors], observations)
        print("slopes:", slopes)
        faulty_segments = [i for i, s in enumerate(slopes) if i > 0 and s < slopes[i-1]]
        while len(faulty_segments) > 0 and base_evaluations < num_evaluations and time.time() < deadline:
            reevaluate = []
            for i in faulty_segments:
                reevaluate = reevaluate + [i-1, i, i+1]
            reevaluate = list(np.unique(reevaluate))
            for i in reevaluate:
                if len(observations[i]) < repeats[i] and time.time() < deadline:
                    print("Re-Evaluting", 2**anchors[i])
                    new_score = evaluate(learner, X, y, 2**anchors[i])
                    observations[i].append(new_score)
                    mean_prev = mean_observations[i]
                    mean_observations[i] = mean(observations[i])
                    print("Updated mean", mean_observations[i], "previously was",mean_prev)
            slopes = getSlopes([2**e for e in anchors], observations)
            print("slopes:", slopes)
            faulty_segments = [i for i, s in enumerate(slopes) if i > 0 and s < slopes[i-1]]
            base_evaluations += 1
        
        #for i, obs in enumerate(slopes):
        
        if len(faulty_segments) == 0:
            print("Obtained stable result for stage " + str(exp))
        else:
            print("Stopped stage " + str(exp) + " with unstable result.")
#        plt.plot(variances)
        
        
        
        # get projections
        if len(slopes) > 0 and stage_id + 1 >= min_stages:
            
            ## convex bound
            last_negative_slope_index = len(slopes) - 1
            while slopes[last_negative_slope_index] > 0:
                last_negative_slope_index -= 1
            slope = slopes[last_negative_slope_index]
            projected_score = mean_observations[stage_id] + (target_size - num_examples) * slope
            print("Projected score for target size " + str(target_size) + " (from anchor " + str(num_examples) + " with mean " + str(mean_observations[stage_id]) + " and slope " + str(slope) + "):", projected_score)
            if projected_score > r:
                print("Impossibly competitive, stopping execution.")
                return np.mean(observations[-1]), anchors, observations
        
            ## inverse power law approximation
            indices = [i for i, exp in enumerate(anchors[:len(mean_observations)]) if exp >= 3][-4:]
            if len(indices) >= 3:
                sizes = np.array([2**e for e in anchors])[indices]
                scores =  np.array(mean_observations)[indices]
                pl_approx = getLCApproximation(sizes, scores)
                #fig, ax = plt.subplots()
                #ax.plot(sizes, scores)
                #domain = np.linspace(0, 10000, 100)
                #ax.plot(domain, pl_approx(domain))
                #plt.show()
    return np.mean(observations[-1]), anchors, observations