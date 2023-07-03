# Learning Curve Cross-validation (LCCV)

## What is it?
Cross-validation (CV) methods such as leave-one-out cross-validation, k-fold cross-validation, and Monte-Carlo cross-validation estimate the predictive performance of a learner by repeatedly training it on a large portion of the given data and testing on the remaining data.
These techniques have two drawbacks.

First, they can be unnecessarily slow on large datasets.
Second, providing only point estimates, they give almost no insights into the learning process of the validated algorithm.
In this repository, we propose a new approach for validation based on learning curves (LCCV).
Instead of creating train-test splits with a large portion of training data, LCCV iteratively increases the number of training examples used for training.
In the context of model selection, it eliminates models that can be safely dismissed from the candidate pool.
We run a large scale experiment on the 67 datasets from the AutoML benchmark, and empirically show that LCCV in over 90% of the cases results in similar performance (at most 0.5% difference) as 10-fold CV, but provides additional insights on the behaviour of a given model.
On top of this, LCCV achieves runtime reductions between 20% and over 50% on half of the 67 datasets from the AutoML benchmark.
This can be incorporated in various AutoML frameworks, to speed up the internal evaluation of candidate models. 
As such, these results can be used orthogonally to other advances in the field of AutoML.

## Usage
You can install LCCV via
```bash
pip install lccv
```

[example-usage.ipynb](https://github.com/fmohr/lccv/blob/master/example-usage.ipynb) shows a Python notebook with usage examples. It also shows how LCCV can be combined with existing AutoML tools like [Naive AutoML](https://github.com/fmohr/naiveautoml).

## Citing LCCV

If you use LCCV in a scientific publication, we would appreciate a reference to
the following paper:

[Felix Mohr and Jan N. van Rijn<br/>
**Fast and Informative Model Selection Using Learning Curve Cross-Validation**<br/>
*IEEE Transactions on Pattern Analysis and Machine Intelligence*](https://www.computer.org/csdl/journal/tp/2023/08/10064171/1LlCTJPbAek)

Bibtex entry:
```bibtex
@article{lccv,
author = {F. Mohr and J. N. van Rijn},
journal = {IEEE Transactions on Pattern Analysis &amp; Machine Intelligence},
title = {Fast and Informative Model Selection Using Learning Curve Cross-Validation},
year = {2023},
volume = {45},
number = {08},
issn = {1939-3539},
pages = {9669-9680},
doi = {10.1109/TPAMI.2023.3251957},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {aug}
}
```

Note that the supplement material can be downloaded on the above link in the *Web Extra* menu.
