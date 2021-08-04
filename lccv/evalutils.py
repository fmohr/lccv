import numpy as np
import pandas as pd
import openml

from func_timeout import func_timeout, FunctionTimedOut

import time
import random

import itertools as it
from scipy.sparse import lil_matrix

import sklearn
from sklearn import metrics
from sklearn import *

def evaluate(learner_inst, X, y, num_examples, seed=0, timeout = None, verbose=False):
    deadline = None if timeout is None else time.time() + timeout
    random.seed(seed)
    n = X.shape[0]
    indices_train = random.sample(range(n), num_examples)
    mask_train = np.zeros(n)
    mask_train[indices_train] = 1
    mask_train = mask_train.astype(bool)
    mask_test = (1 - mask_train).astype(bool)
    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[mask_test][:10000]
    y_test = y[mask_test][:10000]

    if verbose:
        print("Training " + str(learner_inst) + " on data of shape " + str(X_train.shape) + " using seed " + str(seed))
    if deadline is None:
        learner_inst.fit(X_train, y_train)
    else:
        func_timeout(deadline - time.time(), learner_inst.fit, (X_train, y_train))
            
    
    y_hat = learner_inst.predict(X_test)
    error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    if verbose:
        print("Training ready. Obtaining predictions for " + str(X_test.shape[0]) + " instances. Error rate of model on " + str(len(y_hat)) + " instances is " + str(error_rate))
    return error_rate

'''
    Reads in a dataset from openml.org via the ID, returning a matrix X and a label vector y.
    Discrete datasets are checked prior to dummy encoding on whether the encoding should be sparse.
'''
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

'''
   Conducts a 90/10 MCCV (imitating a bit a 10-fold cross validation)
'''
def mccv(learner, X, y, target_size=None, r = 0.0, min_stages = 3, timeout=None, seed=0, repeats = 10):
    
    print("Running mccv with seed " + str(seed))
    train_size = 0.9
    if not timeout is None:
        deadline = time.time() + timeout
    
    scores = []
    n = X.shape[0]
    num_examples = int(train_size * n)
    
    seed *= 13
    for r in range(repeats):
        print("Seed in MCCV:",seed)
        if timeout is None:
            scores.append(evaluate(learner, X, y, num_examples, seed))
        else:
            try:
                if deadline <= time.time():
                    break
                scores.append(func_timeout(deadline - time.time(), evaluate, (learner, X, y, num_examples, seed)))
            except FunctionTimedOut:
                break

            except KeyboardInterrupt:
                break
        seed += 1

    return np.mean(scores) if len(scores) > 0 else np.nan, scores

def select_model(validation, learners, X, y, timeout_per_evaluation, epsilon, seed=0, exception_on_failure=False):
    validation_func = validation[0]
    validation_result_extractor = validation[1]
    
    hard_cutoff = 2 * timeout_per_evaluation
    r = 1.0
    best_score = 1
    chosen_learner = None
    validation_times = []
    for i, learner in enumerate(learners):
        print("\n--------------------------------------------------Checking learner " + str(i + 1) + "/" + str(len(learners)) + " (" + str(learner) + ")\n--------------------------------------------------")
        try:
            validation_start = time.time()
            score = validation_result_extractor(validation_func(learner, X, y, r = r, timeout=timeout_per_evaluation, seed=seed))
            runtime = time.time() - validation_start
            validation_times.append(runtime)
            print("Observed score " + str(score) + " for " + str(learner) + ". Validation took " + str(int(np.round(runtime * 1000))) + "ms")
            r = min(r, score + epsilon)
            print("r is now:", r)
            if score < best_score:
                best_score = score
                chosen_learner = learner
        except KeyboardInterrupt:
            print("Interrupted, stopping")
            break
        except:
            if exception_on_failure:
                raise
            else:
                print("COULD NOT TRAIN " + str(learner) + " on dataset of shape " + str(X.shape) + ". Aborting.")
    return chosen_learner, validation_times

def evaluate_validators(validators, learners, X, y, timeout_per_evaluation, epsilon, seed=0, repeats=10):
    out = {}
    performances = {}
    for validator, result_parser in validators:
        
        print("-------------------------------\n" + validator.__name__ + " (with seed " + str(seed) + ")\n-------------------------------")
        time_start = time.time()
        chosen_learner = select_model((validator, result_parser), learners, X, y, timeout_per_evaluation, epsilon, seed=seed)[0]
        runtime = int(np.round(time.time() - time_start))
        print("Chosen learner is " + str(chosen_learner) + ". Now computing its definitive performance.")
        if chosen_learner is None:
            out[validator.__name__] = ("n/a", runtime, np.nan)
        else:
            if not str(chosen_learner.steps) in performances:
                performances[str(chosen_learner.steps)] = mccv(chosen_learner, X, y, repeats=repeats, seed=4711)
            out[validator.__name__] = (chosen_learner.steps, runtime, performances[str(chosen_learner.steps)])
    return out