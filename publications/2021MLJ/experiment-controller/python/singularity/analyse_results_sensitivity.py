import argparse
import json
import logging
import os
import pandas as pd


def parse_args():
    default_path = '~/experiments/lccv_sensitivity/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=os.path.expanduser('~/results.csv'))
    parser.add_argument('--dataset_id', type=int, default=None)
    parser.add_argument('--hyperparameter', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/lccv_sensitivity'))
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def run(args):
    df = pd.read_csv(args.result_file)

    hyperparameters = df['hyperparameter_name'].unique()
    if args.hyperparameter is not None:
        hyperparameters = args.hyperparameter
    for hyperparameter in hyperparameters:
        frame_hyperparameters = df.loc[df['hyperparameter_name'] == hyperparameter]
        dids = df['dataset_id'].unique()
        if args.dataset_id is not None:
            dids = [args.dataset_id]
        for did in dids:
            frame_did = frame_hyperparameters.loc[frame_hyperparameters['dataset_id'] == did]
            frame_did = frame_did[['hyperparameter_value', 'error_rate', 'runtime']].groupby('hyperparameter_value')
            print("=== %d === %s ===" % (did, hyperparameter))
            print(frame_did.mean())


if __name__ == '__main__':
    args = parse_args()
    run(args)
