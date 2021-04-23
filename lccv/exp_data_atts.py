import argparse
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
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--study_id', type=str, default='OpenML100')

    return parser.parse_args()


def highest_2power_below(n) -> int:
    p = int(np.log2(n))
    return int(pow(2, p))


def run(args):
    suite = openml.study.get_suite(args.study_id)
    task = openml.tasks.get_task(suite.tasks[args.task_idx])
    if not isinstance(task, openml.tasks.OpenMLSupervisedTask):
        raise ValueError('Can run on Supervised Classification tasks')
    x, y = task.get_X_and_y(dataset_format='array')
    size_big = highest_2power_below(len(x))
    indices_big = np.random.permutation(np.arange(len(x)))[:size_big]
    indices_small = indices_big[:int(size_big/2)]
    x_big, y_big = x[indices_big], y[indices_big]
    x_small, y_small = x[indices_small], y[indices_small]

    logging.info('dataset: %s, shape: %s > %s > %s' % (task.get_dataset().name,
                                                       x.shape, x_big.shape,
                                                       x_small.shape))
    clf = sklearn.ensemble.RandomForestClassifier()
    results_small = lccv.lccv(clf, x_small, y_small, verbose=True)
    prediction = results_small[3].get_ipl_estimate_at_target(size_big)

    results_big = sklearn.model_selection.cross_val_score(clf, x, y, cv=10,
                                                          scoring='accuracy')
    print(results_big, prediction)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')
    run(parse_args())
