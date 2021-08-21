import numpy as np
import pandas as pd
import openml
import lccv
import os, psutil
import gc
import logging
import traceback

from func_timeout import func_timeout, FunctionTimedOut

import time
import random

import itertools as it
from scipy.sparse import lil_matrix

import sklearn
from sklearn import metrics
from sklearn import *

from func_timeout import func_timeout, FunctionTimedOut
from commons import *

eval_logger = logging.getLogger("evalutils")

def get_dataset(openmlid):
    """
    Reads in a dataset from openml.org via the ID, returning a matrix X and a label vector y.
    Discrete datasets are checked prior to dummy encoding on whether the encoding should be sparse.
    """
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
        eval_logger.info("creating SPARSE data")
        dfSparse = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], sparse=True)
        
        eval_logger.info("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        eval_logger.info("Done. shape is" + str(X.shape))
    return X, y


def cv10(learner_inst, X, y, timeout=None, seed=None, r=None):
    return cv(learner_inst, X, y, 10, timeout, seed)

def cv5(learner_inst, X, y, timeout=None, seed=None, r=None):
    return cv(learner_inst, X, y, 5, timeout, seed)
    
def cv(learner_inst, X, y, folds, timeout, seed):
    kf = sklearn.model_selection.KFold(n_splits=folds, random_state=np.random.RandomState(seed), shuffle=True)
    scores = []
    deadline = time.time() + timeout if timeout is not None else None
    for train_index, test_index in kf.split(X):
        learner_inst_copy = sklearn.base.clone(learner_inst)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        try:
            
            if timeout is None:
                eval_logger.info(f"Fitting model with {X_train.shape[0]} instances and without timeout.")
                learner_inst_copy.fit(X_train, y_train)
            else:
                timeout_loc = deadline - time.time()
                eval_logger.info(f"Fitting model with {X_train.shape[0]} instances and timeout {timeout_loc}.")
                func_timeout(timeout_loc, learner_inst_copy.fit, (X_train, y_train))
                
            y_hat = learner_inst_copy.predict(X_test)
            error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
            eval_logger.info(f"Observed an error rate of {error_rate}")
            scores.append(error_rate)
        except FunctionTimedOut:
            eval_logger.info(f"Timeout observed for {folds}CV, stopping and using avg of {len(scores)} folds.")
            break
        except KeyboardInterrupt:
            raise
        except:
            eval_logger.info("Observed some exception. Stopping")
            break
    out = np.mean(scores) if scores else np.nan
    eval_logger.info(f"Returning {out} as the avg over observed scores {scores}")
    return out

def lccv90(learner_inst, X, y, r=1.0, timeout=None, seed=None): # maximum train size is 90% of the data (like for 10CV)
    try:
        enforce_all_anchor_evaluations = r == 1
        return lccv.lccv(learner_inst, X, y, r=r, timeout=timeout, seed=seed, target_anchor=.9, min_evals_for_stability=3, MAX_EVALUATIONS = 10, enforce_all_anchor_evaluations = enforce_all_anchor_evaluations,fix_train_test_folds=True)
    except KeyboardInterrupt:
        raise
    except:
        eval_logger.info("Observed some exception. Returning nan")
        return (np.nan,)

def lccv80(learner_inst, X, y, r=1.0, seed=None, timeout=None): # maximum train size is 80% of the data (like for 5CV)
    try:
        enforce_all_anchor_evaluations = r == 1
        return lccv.lccv(learner_inst, X, y, r=r, timeout=timeout, seed=seed, target_anchor=.8, min_evals_for_stability=3, MAX_EVALUATIONS = 5, enforce_all_anchor_evaluations = enforce_all_anchor_evaluations,fix_train_test_folds=True)
    except KeyboardInterrupt:
        raise
    except:
        eval_logger.info("Observed some exception. Returning nan")
        return (np.nan,)

def lccv90flex(learner_inst, X, y, r=1.0, timeout=None, seed=None, **kwargs): # maximum train size is 90% of the data (like for 10CV)
    try:
        enforce_all_anchor_evaluations = r == 1
        return lccv.lccv(learner_inst, X, y, r=r, timeout=timeout, seed=seed, target_anchor=.9, min_evals_for_stability=3, MAX_EVALUATIONS = 10, enforce_all_anchor_evaluations = enforce_all_anchor_evaluations,fix_train_test_folds=False, **kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        traceback.print_exc()
        eval_logger.info("Observed some exception. Returning nan")
        return (np.nan,)

def lccv80flex(learner_inst, X, y, r=1.0, seed=None, timeout=None): # maximum train size is 80% of the data (like for 5CV)
    try:
        enforce_all_anchor_evaluations = r == 1
        return lccv.lccv(learner_inst, X, y, r=r, timeout=timeout, seed=seed, target_anchor=.8, min_evals_for_stability=3, MAX_EVALUATIONS = 5, enforce_all_anchor_evaluations = enforce_all_anchor_evaluations,fix_train_test_folds=False)
    except KeyboardInterrupt:
        raise
    except:
        eval_logger.info("Observed some exception. Returning nan")
        return (np.nan,)

def mccv(learner, X, y, target_size=.9, r = 0.0, min_stages = 3, timeout=None, seed=0, repeats = 10):
    
    def evaluate(learner_inst, X, y, num_examples, seed=0, timeout = None, verbose=False):
        deadline = None if timeout is None else time.time() + timeout
        random.seed(seed)
        n = X.shape[0]
        indices_train = random.sample(range(n), num_examples)
        mask_train = np.zeros(n)
        mask_train[indices_train] = 1
        mask_train = mask_train.astype(bool)
        mask_test = (1 - mask_train).astype(bool)
        X_train = X[mask_train].copy()
        y_train = y[mask_train]
        X_test = X[mask_test].copy()
        y_test = y[mask_test]
        learner_inst = sklearn.base.clone(learner_inst)

        eval_logger.info(f"Training {format_learner(learner_inst)} on data of shape {X_train.shape} using seed {seed}.")
        if deadline is None:
            learner_inst.fit(X_train, y_train)
        else:
            func_timeout(deadline - time.time(), learner_inst.fit, (X_train, y_train))


        y_hat = learner_inst.predict(X_test)
        error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
        eval_logger.info(f"Training ready. Obtaining predictions for {X_test.shape[0]} instances. Error rate of model on {len(y_hat)} instances is {error_rate}")
        return error_rate
    
    """
    Conducts a 90/10 MCCV (imitating a bit a 10-fold cross validation)
    """
    eval_logger.info(f"Running mccv with seed  {seed}")
    if not timeout is None:
        deadline = time.time() + timeout
    
    scores = []
    n = X.shape[0]
    num_examples = int(target_size * n)
    
    seed *= 13
    for r in range(repeats):
        eval_logger.info(f"Seed in MCCV: {seed}. Training on {num_examples} examples. That is {np.round(100 * num_examples / X.shape[0])}% of the data (testing on rest).")
        if timeout is None:
            try:
                scores.append(evaluate(learner, X, y, num_examples, seed))
            except KeyboardInterrupt:
                raise
                
            except:
                eval_logger.info("AN ERROR OCCURRED, not counting this run!")
        else:
            try:
                if deadline <= time.time():
                    break
                scores.append(func_timeout(deadline - time.time(), evaluate, (learner, X, y, num_examples, seed)))
            except FunctionTimedOut:
                break

            except KeyboardInterrupt:
                raise
                
            except:
                eval_logger.info("AN ERROR OCCURRED, not counting this run!")
        seed += 1

    return np.mean(scores) if len(scores) > 0 else np.nan, scores

def format_learner(learner):
    learner_name = str(learner).replace("\n", " ").replace("\t", " ")
    for k in  range(20):
        learner_name = learner_name.replace("  ", " ")
    return learner_name

def select_model(validation, learners, X, y, timeout_per_evaluation, epsilon, seed=0, exception_on_failure=False):
    validation_func = validation[0]
    validation_result_extractor = validation[1]
    kwargs = {}
    if len(validation) > 2:
        kwargs = validation[2]
        eval_logger.info("Running with additional arguments: %s" % str(kwargs))
    
    hard_cutoff = 2 * timeout_per_evaluation
    r = 1.0
    best_score = 1
    chosen_learner = None
    validation_times = []
    exp_logger = logging.getLogger("experimenter")
    n = len(learners)
    memory_history = []
    index_of_best_learner = -1
    for i, learner in enumerate(learners):
        exp_logger.info(f"""
            --------------------------------------------------
            Checking learner {i + 1}/{n} ({format_learner(learner)})
            --------------------------------------------------""")
        cur_mem = int(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        memory_history.append(cur_mem)
        exp_logger.info(f"Currently used memory: {cur_mem}MB. Memory history is: {memory_history}")
        try:
            validation_start = time.time()
            temp_pipe = sklearn.pipeline.Pipeline([(step_name, build_estimator(comp, params, X, y)) for step_name, (comp, params) in learner])
            score = validation_result_extractor(validation_func(temp_pipe, X, y, r = r, timeout=timeout_per_evaluation, seed=13 *seed + i, **kwargs))
            runtime = time.time() - validation_start
            validation_times.append(runtime)
            eval_logger.info(f"Observed score {score} for {format_learner(temp_pipe)}. Validation took {int(np.round(runtime * 1000))}ms")
            r = min(r, score + epsilon)
            eval_logger.info(f"r is now: {r}")
            if score < best_score:
                best_score = score
                chosen_learner = temp_pipe
                index_of_best_learner = i
                eval_logger.info(f"Thas was a NEW BEST score. r has been updated. In other words, currently chosen model is {format_learner(chosen_learner)}")
            else:
                del temp_pipe
                gc.collect()
                eval_logger.info(f"Candidate was NOT competitive. Eliminating the object and garbage collecting.")

        except KeyboardInterrupt:
            eval_logger.warning("Interrupted, stopping")
            break
        except:
            if True or exception_on_failure:
                raise
            else:
                eval_logger.warning(f"COULD NOT TRAIN {learner} on dataset of shape {X.shape}. Aborting.")
    eval_logger.info(f"Chosen learner was found in iteration {index_of_best_learner + 1}")
    return chosen_learner, validation_times

def evaluate_validators(validators, learners, X, y, timeout_per_evaluation, epsilon, seed=0, repeats=10):
    out = {}
    performances = {}
    for validator in validators:
        
        eval_logger.info(f"""
        -------------------------------
        {validator[0].__name__} (with seed {seed})
        -------------------------------""")
        time_start = time.time()
        chosen_learner = select_model(validator, learners, X, y, timeout_per_evaluation, epsilon, seed=seed)[0]
        runtime = int(np.round(time.time() - time_start))
        eval_logger.info("Chosen learner is " + str(chosen_learner) + ". Now computing its definitive performance.")
        if chosen_learner is None:
            out[validator[0].__name__] = ("n/a", runtime, np.nan)
        else:
            if not str(chosen_learner.steps) in performances:
                if validator[0].__name__ in ["cv5", "lccv80", "lccv80flex"]:
                    target_size = .8
                elif validator[0].__name__ in ["cv10", "lccv90", "lccv90flex"]:
                    target_size = .9
                else:
                    raise Exception(
                        f"Invalid validator function {validator.__name__}")
                eval_logger.info(f"Appplying target size {target_size}")
                performances[str(chosen_learner.steps)] = mccv(chosen_learner, X, y, target_size = target_size, repeats=repeats, seed=4711)
            out[validator[0].__name__] = (chosen_learner.steps, runtime, performances[str(chosen_learner.steps)])
    return out