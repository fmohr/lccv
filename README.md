# Learning Curve Cross-validation (LCCV)

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

## Citing LCCV

If you use LCCV in a scientific publication, we would appreciate a reference to
the following paper:

[Felix Mohr and Jan N. van Rijn<br/>
**Towards Model Selection using Learning Curve Cross-Validation**<br/>
*8th ICML Workshop on Automated Machine Learning (AutoML)*](https://openreview.net/attachment?id=EC_IHbAaMG&name=crc_pdf)

Bibtex entry:
```bibtex
@inproceedings{mohr2021towards,
  author = {Mohr, Felix and and van Rijn, Jan N.},
  title = {Towards Model Selection using Learning Curve Cross-Validation},
  booktitle = {8th ICML Workshop on Automated Machine Learning (AutoML)},
  year = {2021}
}
```

We are actively revising a version of this paper for a rigorously peer-reviewed journal.

