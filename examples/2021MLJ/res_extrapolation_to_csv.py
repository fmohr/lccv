import argparse
import json
import logging
import os
import pandas as pd


from exp_data_atts import learners


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default=os.path.expanduser('~/experiments/lccv'))
    parser.add_argument('--output_file', type=str,
                        default='data/extrapolation.csv')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')
    args = parse_args()

    decode_errors = 0
    all_rows = []
    for task_dir in os.listdir(args.input_dir):
        for learner in learners:
            filename = 'result_' + str(learner) + '.json'
            filepath = os.path.join(args.input_dir, task_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as fp:
                    try:
                        task_results = json.load(fp)
                        lccv_size = max(task_results['lccv'].keys())
                        current = task_results['lccv'][lccv_size]
                        prediction = task_results['prediction'][str(task_results['sizes'][1])]
                        actual = task_results['cv'][str(task_results['sizes'][1])]

                        all_rows.append({
                            'task_id': int(task_dir),
                            'classifier': str(learner),
                            'performance_curve_end': current['mean'],
                            'performance_prediction': prediction,
                            'performance_next_point': actual['mean'],
                            'delta_current_prediction': current['mean'] - prediction,
                            'delta_current_actual': current['mean'] - actual['mean'],
                        })
                    except json.decoder.JSONDecodeError as e:
                        logging.warning('JSON decode error for file: %s' % filepath)
                        decode_errors += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_file)
    n_tasks = len(os.listdir(args.input_dir))
    logging.info('tasks: %d, learners: %d, results: %d, results/task: %f, decode errors: %d' %
                 (n_tasks, len(learners), len(df), len(df) / n_tasks, decode_errors))
