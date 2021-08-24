import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=('../../../../../data/sensitivity.csv'))
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/lccv_sensitivity'))
    parser.add_argument('--dimension1', type=str, default='hyperparameter_name')
    parser.add_argument('--extension', type=str, default='png')
    return parser.parse_args()


def run(args):
    df = pd.read_csv(args.result_file)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dimension1 == 'dataset_id':
        dimension1 = 'dataset_id'
        dimension2 = 'hyperparameter_name'
    elif args.dimension1 == 'hyperparameter_name':
        dimension1 = 'hyperparameter_name'
        dimension2 = 'dataset_id'
    else:
        raise ValueError()

    dimension1_unique_vals = df[dimension1].unique()

    for dimension1_value in dimension1_unique_vals:
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        frame_sliced = df.loc[df[dimension1] == dimension1_value]
        dimension2_unique_vals = df[dimension2].unique()

        for dimension2_value in dimension2_unique_vals:
            frame_double_sliced = frame_sliced.loc[frame_sliced[dimension2] == dimension2_value]
            try:
                frame_double_sliced.loc[:, 'hyperparameter_value'] = frame_double_sliced['hyperparameter_value'].astype(np.int)
            except ValueError:
                try:
                    frame_double_sliced.loc[:, 'hyperparameter_value'] = frame_double_sliced['hyperparameter_value'].astype(np.float64)
                except ValueError:
                    pass
            frame_double_sliced = frame_double_sliced[['hyperparameter_value', 'error_rate', 'runtime']].groupby('hyperparameter_value')
            frame_mean = frame_double_sliced.mean()
            values = frame_mean.index.to_numpy()
            runtimes = frame_mean['runtime'].to_numpy()
            error_rates = frame_mean['error_rate'].to_numpy()
            ax.plot(runtimes, error_rates, label=dimension2_value)
            for i in range(len(values)):
                ax.annotate(values[i], (runtimes[i], error_rates[i]))
            print("=== %s === %s ===" % (dimension1_value, dimension2_value))
            print(frame_double_sliced.mean())
        filename = os.path.join(args.output_dir, '%s.%s' % (dimension1_value, args.extension))
        plt.legend()
        plt.savefig(filename)
        plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    run(args)
