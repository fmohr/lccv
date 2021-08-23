import argparse
import json
import logging
import os
import pandas as pd


def parse_args():
    default_path = '~/experiments/lccv_sensitivity/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=os.path.expanduser(default_path))
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def run(args):

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    count = 0
    records = []

    for dataset_id in os.listdir(args.results_dir):
        dataset_dir = os.path.join(args.results_dir, dataset_id)
        if os.path.isfile(dataset_dir):
            continue
        logging.debug('- dataset dir: %s' % dataset_id)
        for hyperparameter_name in os.listdir(dataset_dir):
            hpname_dir = os.path.join(args.results_dir, dataset_id, hyperparameter_name)
            logging.debug('-- hyperparameter name: %s' % hpname_dir)
            for hyperparameter_value in os.listdir(hpname_dir):
                hpvalue_dir = os.path.join(args.results_dir, dataset_id, hyperparameter_name, hyperparameter_value)
                logging.debug('--- hyperparameter value: %s' % hpvalue_dir)
                for seed in os.listdir(hpvalue_dir):
                    file = os.path.join(args.results_dir, dataset_id, hyperparameter_name, hyperparameter_value, seed, 'results.txt')
                    logging.debug('---> file: %s' % file)
                    if os.path.isfile(file):
                        with open(file, 'r') as fp:
                            result = json.load(fp)
                            record = {
                                'dataset_id': int(dataset_id),
                                'hyperparameter_name': str(hyperparameter_name),
                                'hyperparameter_value': str(hyperparameter_value),
                                'seed': int(seed),
                                'error_rate': float(result[1]),
                                'runtime': float(result[2])
                            }
                            records.append(record)
                        # print(file)
                        count += 1

    logging.info('total files found: %d' % count)
    if count > 0:
        frame = pd.DataFrame(record)
        result_file = os.path.join(args.results_dir), 'results.csv'
        frame.to_csv(result_file)
        logging.info('results saved to: %s' % result_file)


if __name__ == '__main__':
    args = parse_args()
    run(args)
