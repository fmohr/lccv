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
        dataset_dir = os.path.join(directory, dataset_id)
        print('- dataset dir:', dataset_id)
        for hyperparameter_name in os.listdir(dataset_dir):
            hpname_dir = os.path.join(directory, dataset_id, hyperparameter_name)
            print('-- hyperparameter name:', hpname_dir)
            for hyperparameter_value in os.listdir(hpname_dir):
                hpvalue_dir = os.path.join(directory, dataset_id, hyperparameter_name, hyperparameter_value)
                print('--- hyperparameter value:', hpname_dir)
                for seed in os.listdir(hpvalue_dir):
                    file = os.path.join(directory, dataset_id, hyperparameter_name, hyperparameter_value, seed, 'results.txt')
                    print('---> file:', file)

                    if os.path.isfile(file):
                        print(file)
                        count += 1
    print('total count', count)


if __name__ == '__main__':
    args = parse_args()
    run(args.output_dir)
