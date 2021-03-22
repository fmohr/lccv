import numpy as np
import pandas as pd
import openml

from func_timeout import func_timeout, FunctionTimedOut

import time
import random

import itertools as it
from scipy.sparse import lil_matrix

from tqdm.notebook import tqdm

import sklearn
from sklearn import metrics
from sklearn import *

learners = [
    (sklearn.svm.LinearSVC, {}),
    (sklearn.tree.DecisionTreeClassifier, {}),
    (sklearn.tree.ExtraTreeClassifier, {}),
    (sklearn.linear_model.LogisticRegression, {}),
    (sklearn.linear_model.PassiveAggressiveClassifier, {}),
    (sklearn.linear_model.Perceptron, {}),
    (sklearn.linear_model.RidgeClassifier, {}),
    (sklearn.linear_model.SGDClassifier, {}),
    (sklearn.neural_network.MLPClassifier, {}),
    (sklearn.discriminant_analysis.LinearDiscriminantAnalysis, {}),
    (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis, {}),
    (sklearn.naive_bayes.BernoulliNB, {}),
    (sklearn.naive_bayes.MultinomialNB, {}),
    (sklearn.neighbors.KNeighborsClassifier, {}),
    (sklearn.ensemble.ExtraTreesClassifier, {}),
    (sklearn.ensemble.RandomForestClassifier, {}),
    (sklearn.ensemble.GradientBoostingClassifier, {})
]

def evaluate(learner, X, y, num_examples, seed=0):
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

    inst = learner()
    #print("Training " + str(learner) + " on data of shape " + str(X_train.shape))
    inst.fit(X_train, y_train)
    #print("Training ready. Obtaining predictions for " + str(X_test.shape[0]) + " instances.")
    y_hat = inst.predict(X_test)
    return 1 - sklearn.metrics.accuracy_score(y_test, y_hat)

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
def mccv(learner, X, y, target_size=None, r = 0.0, min_stages = 3, timeout=60, seed=0):

    train_size = 0.9
    repeats = 10
    deadline = time.time() + timeout
    
    scores = []
    n = X.shape[0]
    num_examples = int(train_size * n)
    
    for r in range(repeats):
        try:
            if deadline <= time.time():
                break
            scores.append(func_timeout(deadline - time.time(), evaluate, (learner, X, y, num_examples, seed+1)))
        except FunctionTimedOut:
            break
            
        except KeyboardInterrupt:
            break
    
    return np.mean(scores) if len(scores) > 0 else np.nan, scores

def select_model(validation, learners, X, y, timeout_per_evaluation, epsilon, exception_on_failure=False):
    validation_func = validation[0]
    validation_result_extractor = validation[1]
    
    hard_cutoff = 2 * timeout_per_evaluation
    r = 1.0
    best_score = 1
    chosen_learner = None
    validation_times = []
    for learner in tqdm(learners):
        print("Checking learner " + str(learner))
        try:
            validation_start = time.time()
            score = validation_result_extractor(validation_func(learner, X, y, r = r, timeout=timeout_per_evaluation))
            validation_times.append(time.time() - validation_start)
            print("Observed score " + str(score) + " for " + str(learner))
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

def evaluate_validator(validation, learners, X, y, timeout_per_evaluation, epsilon):
    chosen_learner = select_model(validation, learners, X, y, timeout_per_evaluation, epsilon)
    print("Chosen learner is " + str(chosen_learner) + ". Now computing its definitive performance.")
    true_performance = cv10(chosen_learner, X, y, timeout=60*60*24, seed=0)
    return true_performance