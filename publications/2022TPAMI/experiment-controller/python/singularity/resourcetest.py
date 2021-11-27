import json
import ConfigSpace
from ConfigSpace.util import *
from ConfigSpace.read_and_write import json as config_json
import random
from commons import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier
import os

print(os.environ["OMP_NUM_THREADS"])
print(os.environ["MKL_NUM_THREADS"])
print(os.environ["OPENBLAS_NUM_THREADS"])
print(os.environ["BLIS_NUM_THREADS"])

#X, y = get_dataset(1485)
X, y = get_dataset(1485)
y = y.to_numpy()
for i in range(6):
    print(X.shape)
    print(y.shape)
    X = np.row_stack([X, X])
    y = np.row_stack([y.reshape(len(y), 1), y.reshape(len(y), 1)])
    print(X.shape, y.shape)

pl = Pipeline(steps=[('predictor', MLPClassifier())])
pl.fit(X, y)