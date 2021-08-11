import argparse
import logging
import openml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_id', type=str, default=271)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')
    args = parse_args()
    suite = openml.study.get_suite(args.study_id)
    for task_idx, task_id in enumerate(suite.tasks):
        logging.info('(%d/%d) starting task %d' % (
            task_idx + 1, len(suite.tasks), task_id))
        # automatically stores all relevant info in .cache
        task = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
