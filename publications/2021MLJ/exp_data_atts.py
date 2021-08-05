import argparse
import json
import logging
import numpy as np
import os

import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.impute
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

import lccv

import openml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/lccv/')
    parser.add_argument('--job_idx', type=int, default=None)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--study_id', type=str, default=271)

    return parser.parse_args()


learners = [
    sklearn.svm.LinearSVC(),
    sklearn.tree.DecisionTreeClassifier(),
    sklearn.tree.ExtraTreeClassifier(),
    sklearn.linear_model.LogisticRegression(),
    sklearn.linear_model.PassiveAggressiveClassifier(),
    sklearn.linear_model.Perceptron(),
    sklearn.linear_model.RidgeClassifier(),
    sklearn.linear_model.SGDClassifier(),
    sklearn.neural_network.MLPClassifier(),
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
    sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),
    sklearn.naive_bayes.BernoulliNB(),
    sklearn.naive_bayes.MultinomialNB(),
    sklearn.neighbors.KNeighborsClassifier(),
    sklearn.ensemble.ExtraTreesClassifier(),
    sklearn.ensemble.RandomForestClassifier(),
    sklearn.ensemble.GradientBoostingClassifier(),
]


def highest_2power_below(n) -> int:
    p = int(np.log2(n * .9))
    return int(pow(2, p) / 9 * 10)


def clf_as_pipeline(clf, numeric_indices, nominal_indices):
        numeric_transformer = sklearn.pipeline.make_pipeline(
            sklearn.impute.SimpleImputer(),
            sklearn.preprocessing.StandardScaler())

        # note that the dataset is encoded numerically, hence we can only impute
        # numeric values, even for the categorical columns.
        categorical_transformer = sklearn.pipeline.make_pipeline(
            sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1),
            sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))

        transformer = sklearn.compose.ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_indices),
                ('nominal', categorical_transformer, nominal_indices)],
            remainder='passthrough')

        pipeline = sklearn.pipeline.make_pipeline(transformer,
                                                  sklearn.feature_selection.VarianceThreshold(),
                                                  clf)
        return pipeline


def run_classifier_on_task(
        learner_idx: int,
        task: openml.tasks.OpenMLSupervisedTask,
        output_directory: str, verbose: bool):
    nominal_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
    numeric_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])
    clf = clf_as_pipeline(learners[learner_idx], numeric_indices, nominal_indices)
    x, y = task.get_X_and_y(dataset_format='array')
    unique, counts = np.unique(y, return_counts=True)
    logging.info('class dist (all): %s' % dict(zip(unique, counts)))
    size_big = highest_2power_below(len(x))

    indices_big = np.random.permutation(np.arange(len(x)))[:size_big]
    indices_small = indices_big[:int(size_big/2)]
    x_big, y_big = x[indices_big], y[indices_big]
    unique_big, counts_big = np.unique(y_big, return_counts=True)
    logging.info('class dist (big): %s' % dict(zip(unique_big, counts_big)))
    x_small, y_small = x[indices_small], y[indices_small]
    unique_small, counts_small = np.unique(y_small, return_counts=True)
    logging.info('class dist (small): %s' % dict(zip(unique_small, counts_small)))

    output_dir = os.path.join(output_directory, str(task.task_id))
    os.makedirs(output_dir, exist_ok=True)
    filename = 'result_%s.json' % str(learners[learner_idx])  # do not use full pipeline name
    if os.path.isfile(os.path.join(output_dir, filename)):
        logging.info('clf %s on dataset %s already exists' % (str(learners[learner_idx]), task.get_dataset().name))
        return

    logging.info('dataset: %s, shape: %s > %s > %s' % (task.get_dataset().name,
                                                       x.shape, x_big.shape,
                                                       x_small.shape))
    results_lccv = lccv.lccv(clf, x_small, y_small,
                             enforce_all_anchor_evaluations=True, verbose=verbose)
    prediction = results_lccv[3].get_ipl_estimate_at_target(size_big)

    cv_big = sklearn.model_selection.cross_val_score(
        clf, x_big, y_big, cv=10, scoring='accuracy')
    cv_small = sklearn.model_selection.cross_val_score(
        clf, x_small, y_small, cv=10, scoring='accuracy')

    all_results = {
        'sizes': [int(size_big/2), size_big],
        'lccv': results_lccv[2],
        'cv': {
            len(x_small): {
                'n': 10,
                'mean': np.mean(1 - cv_small),
                'std': np.std(1 - cv_small)
            },
            len(x_big): {
                'n': 10,
                'mean': np.mean(1 - cv_big),
                'std': np.std(1 - cv_big)
            }
        },
        'prediction': {
            int(size_big): prediction
        }
    }
    with open(os.path.join(output_dir, filename), 'w') as fp:
        json.dump(all_results, fp)
    logging.info('Results written to file: %s' % os.path.join(output_dir, filename))


def run(args):
    suite = openml.study.get_suite(args.study_id)
    if args.job_idx is not None:
        task_idx = int(args.job_idx / len(learners))
        learner_idx = int(args.job_idx % len(learners))
        task = openml.tasks.get_task(suite.tasks[task_idx])
        if not isinstance(task, openml.tasks.OpenMLSupervisedTask):
            raise ValueError('Can run on Supervised Classification tasks')
        run_classifier_on_task(learner_idx, task, args.output_directory, args.verbose)
    else:
        for task_idx, task_id in enumerate(suite.tasks):
            for learner_idx in range(len(learners)):
                try:
                    task = openml.tasks.get_task(task_id)
                    data_name = task.get_dataset().name
                    logging.info('(%d/%d) starting task %d: %s' % (
                        task_idx+1, len(suite.tasks), task.task_id, data_name))
                    if not isinstance(task, openml.tasks.OpenMLSupervisedTask):
                        raise ValueError('Can run on Supervised Classif. tasks')
                    run_classifier_on_task(learner_idx, task, args.output_directory, args.verbose)
                except Exception as e:
                    logging.warning('An exception: %s' % str(e))
                    print(e)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')
    args = parse_args()
    run(args)
