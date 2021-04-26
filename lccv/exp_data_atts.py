import argparse
import json
import logging
import numpy as np
import os

import sklearn.tree

import lccv

import openml.study


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/lccv/')
    parser.add_argument('--task_idx', type=int)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--study_id', type=str, default=218)

    return parser.parse_args()


def highest_2power_below(n) -> int:
    p = int(np.log2(n * .9))
    return int(pow(2, p) / 9 * 10)


def run_on_task(
        task: openml.tasks.OpenMLSupervisedTask,
        output_directory: str, verbose: bool):
    x, y = task.get_X_and_y(dataset_format='array')
    size_big = highest_2power_below(len(x))

    indices_big = np.random.permutation(np.arange(len(x)))[:size_big]
    indices_small = indices_big[:int(size_big/2)]
    x_big, y_big = x[indices_big], y[indices_big]
    x_small, y_small = x[indices_small], y[indices_small]

    output_dir = os.path.join(output_directory, str(task.task_id))
    os.makedirs(output_dir, exist_ok=True)

    logging.info('dataset: %s, shape: %s > %s > %s' % (task.get_dataset().name,
                                                       x.shape, x_big.shape,
                                                       x_small.shape))
    clf = sklearn.ensemble.RandomForestClassifier()
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
    with open(os.path.join(output_dir, 'result.json'), 'w') as fp:
        json.dump(all_results, fp)


def run(args):
    suite = openml.study.get_suite(args.study_id)
    if args.task_idx is not None:
        task = openml.tasks.get_task(suite.tasks[args.task_idx])
        if not isinstance(task, openml.tasks.OpenMLSupervisedTask):
            raise ValueError('Can run on Supervised Classification tasks')
        run_on_task(task, args.output_directory, args.verbose)
    else:
        for idx, task_id in enumerate(suite.tasks):
            try:
                task = openml.tasks.get_task(task_id)
                data_name = task.get_dataset().name
                logging.info('(%d/%d) starting task %d: %s' % (
                    idx+1, len(suite.tasks), task.task_id, data_name))
                if not isinstance(task, openml.tasks.OpenMLSupervisedTask):
                    raise ValueError('Can run on Supervised Classif. tasks')
                run_on_task(task, args.output_directory, args.verbose)
            except Exception as e:
                logging.warning('An exception')
                print(e)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')
    args = parse_args()
    run(args)
