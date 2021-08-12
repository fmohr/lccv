import argparse
import os


def parse_args():
    default_path = '~/experiments/lccv/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser(default_path))
    return parser.parse_args()


def run(directory):
    count = 0
    for dataset_id in os.listdir(directory):
        for algorithm_name in os.listdir(os.path.join(directory, dataset_id)):
            for seed in os.listdir(os.path.join(directory, dataset_id, algorithm_name)):
                file = os.path.join(directory, dataset_id, seed, 'results.txt')
                if os.path.isfile(file):
                    print(file)
                    count += 1
    print('total count', count)


if __name__ == '__main__':
    args = parse_args()
    run(args.output_dir)
