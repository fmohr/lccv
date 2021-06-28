import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='plots/example_curves')
    return parser.parse_args()


def plot_series(axes, output_dir, prefix, curves, colors):
    all_sizes = np.array([64, 128, 256, 1024])
    stdevs = np.array([0.5, 0.25, 0.1, 0.05])

    for curve_idx, current_curve in enumerate(curves):
        for size_idx in range(len(all_sizes)-1):
            current_len = size_idx+2
            current_sizes = all_sizes[0:current_len]
            print(current_sizes, current_curve[0:current_len])
            axes.fill_between(
                current_sizes,
                current_curve[0:current_len] - stdevs[:current_len],
                current_curve[0:current_len] + stdevs[:current_len],
                alpha=0.1, color=colors[curve_idx]
            )
            axes.plot(current_sizes, current_curve[0:size_idx+2],
                      'o-', color=colors[curve_idx])
            filename = os.path.join(output_dir, '%s_%d_%d.pdf' % (prefix, curve_idx, size_idx))
            plt.tight_layout()
            plt.savefig(filename)


def run(output_dir):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=36)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    curves_convex = [
        [10.0, 8.0, 7.0, 6.5],
        [12.0, 9.0, 7.0, 6.0],
        [15.0,11.0, 8.0, 6.25],
    ]

    curves_concave = [
        [12.0, 11.0, 9.0, 8.0],
        [10.0, 8.0, 7.0, 5.0],
        [15.0, 12.0, 12.0, 10.0],
    ]

    for i in range(2):
        axes[i].set_xlabel("Training examples")
        axes[i].set_ylabel("Error rate")
        axes[i].set_xlim([64, 1100])
        axes[i].set_ylim([4, 16])
        axes[i].set_xscale('log')

    axes[0].set_title("Convex learning curves")
    plot_series(axes[0], output_dir, 'convex', curves_convex, ['b', 'g', 'r'])
    axes[1].set_title("Concave learning curves")
    plot_series(axes[1], output_dir, 'concave', curves_concave, ['c', 'm', 'y'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')
    args = parse_args()
    run(args.output_dir)
