import argparse
import json
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=os.path.expanduser('~/results.csv'))
    parser.add_argument('--dataset_id', type=int, default=None)
    parser.add_argument('--hyperparameter', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/lccv_sensitivity'))
    parser.add_argument('--extension', type=str, default='png')
    return parser.parse_args()


def run(args):
    df = pd.read_csv(args.result_file)
    os.makedirs(args.output_dir, exist_ok=True)

    hyperparameters = df['hyperparameter_name'].unique()
    if args.hyperparameter is not None:
        hyperparameters = args.hyperparameter
    for hyperparameter in hyperparameters:
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        frame_hyperparameters = df.loc[df['hyperparameter_name'] == hyperparameter]
        dids = df['dataset_id'].unique()
        if args.dataset_id is not None:
            dids = [args.dataset_id]
        for did in dids:
            frame_did = frame_hyperparameters.loc[frame_hyperparameters['dataset_id'] == did]
            frame_did = frame_did[['hyperparameter_value', 'error_rate', 'runtime']].groupby('hyperparameter_value')
            ax.plot(frame_did.mean()['runtime'].to_numpy(), frame_did.mean()['error_rate'].to_numpy())
            print("=== %d === %s ===" % (did, hyperparameter))
            print(frame_did.mean())
        filename = os.path.join(args.output_dir, '%s.%s' % (hyperparameter, args.extension))
        plt.savefig(filename)
        plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    run(args)
