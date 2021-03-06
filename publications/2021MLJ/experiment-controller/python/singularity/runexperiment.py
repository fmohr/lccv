import argparse
import resource
from evalutils import *
from lccv import lccv
from commons import *
import logging
import json


def parse_args():
    default_path = '~/experiments/lccv/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_idx', type=int)
    parser.add_argument('--num_seeds', type=int)
    parser.add_argument('--dataset_id', type=int)
    parser.add_argument('--algorithm', type=str, choices=['10cv', '5cv', '80lccv', '90lccv', '80lccv-flex', '90lccv-flex'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--num_pipelines', type=int, default=1000)
    parser.add_argument('--folder', type=str, default=os.path.expanduser(default_path))
    parser.add_argument('--prob_dp', type=float, default=0.5)
    parser.add_argument('--prob_fp', type=float, default=0.5)
    return parser.parse_args()


def run_experiment(openmlid: int, algorithm: str, num_pipelines: int, seed: int, timeout: int, folder: str, prob_dp: float, prob_fp: float):
    # TODO: built in check whether file already exists, in that case we can skipp
    
    # CPU
    print("CPU Settings:")
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        print(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")
        
    # memory limits
    memory_limit = 14 * 1024
    print("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    # configure lccv logger (by default set to WARN, change it to DEBUG if tests fail)
    lccv_logger = logging.getLogger("lccv")
    lccv_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    lccv_logger.addHandler(ch)
	
    exp_logger = logging.getLogger("experimenter")
    exp_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    exp_logger.addHandler(ch)
    
    eval_logger = logging.getLogger("evalutils")
    eval_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    eval_logger.addHandler(ch)
    
    exp_logger.info("Starting python script")
    exp_logger.info(f"""Running experiment under following conditions:
    OpenML id: {openmlid}
    Algorithm: {algorithm}
    Seed: {seed}
    timeout (per single evaluation):  {timeout}
    Num Pipelines: {num_pipelines}
    Probability to draw a data-preprocessor: {prob_dp}
    Probability to draw a feature-preprocessor: {prob_fp}
    """)
    
    # CPU
    exp_logger.info("CPU Settings:")
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        exp_logger.info(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")
        
    # memory limits
    memory_limit = 14 * 1024
    exp_logger.info("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    

    
    # load data
    exp_logger.info("Reading dataset")
    X, y = get_dataset(openmlid)
    exp_logger.info(f"ready. Dataset shape is {X.shape}, label column shape is {y.shape}. Now running the algorithm")
    if X.shape[0] <= 0:
        raise Exception("Dataset size invalid!")
    if X.shape[0] != len(y):
        raise Exception("X and y do not have the same size.")
    
    def mccv_adaptive(learner_inst, X, y, timeout, seed=0, r=None):
        return lccv(learner_inst, X, y, r = 1.0, eps = 0, timeout=timeout, base = 2, min_exp = np.log(int(np.floor(X.shape[0] * 0.9))) / np.log(2), MAX_EVALUATIONS = 10, seed=seed, enforce_all_anchor_evaluations=False, verbose=True)
    
    # creating learner sequence
    sampler = PipelineSampler("searchspace.json", X, y, seed, dp_proba = prob_dp, fp_proba = prob_fp)
    test_learners = [sampler.sample(do_build=False) for i in range(num_pipelines)]
    exp_logger.info(f"Evaluating portfolio of {len(test_learners)} learners.")
    
    # run lccv
    epsilon = 0.0
    if algorithm == "5cv":
        validators = validators = [(cv5, lambda r: r)]
        key = "cv5"
    elif algorithm == "10cv":
        validators = validators = [(cv10, lambda r: r)]
        key = "cv10"
    elif algorithm == "90lccv":
        validators = validators = [(lccv90, lambda r: r[0])]
        key = "lccv90"
    elif algorithm == "80lccv":
        validators = validators = [(lccv80, lambda r: r[0])]
        key = "lccv80"    
    elif algorithm == "90lccv-flex":
        validators = validators = [(lccv90flex, lambda r: r[0])]
        key = "lccv90flex"
    elif algorithm == "80lccv-flex":
        validators = validators = [(lccv80flex, lambda r: r[0])]
        key = "lccv80flex"
    else:
        raise Exception(f"Unsupported validation algorithm {algorithm}")
    result = evaluate_validators(validators, test_learners, X, y, timeout, seed=seed, repeats=100, epsilon=epsilon)[key]
    model = result[0]
    runtime = result[1]
    if model is not None:
        error_rate = np.round(result[2][0], 4)
        error_rates = list(np.round(result[2][1], 4))
        model_name = str(model).replace("\n", " ")
        exp_logger.info(f"""Run completed. Here are the details:
            Model: {model}
            Error Rate: {error_rate}
            Runtime: {runtime}
            Results in final evaluation: {np.array(error_rates)}""")
    else:
        exp_logger.info("No model was chosen. Assigning maximum error rate")
        error_rate = 1
        model_name = "None"
    
    # write result
    output = (model_name, error_rate, error_rates, runtime)
    with open(folder + "/results.txt", "w") as outfile: 
        json.dump(output, outfile)
    exp_logger.info(f"Experiment ready. Results written to {folder}/results.txt")


def run_experiment_index_based(index: int, num_seeds: int, algorithm: str, num_pipelines: int, timeout: int, prob_dp: float, prob_fp: float):
    datasets = [
        1485, 1590, 1515, 1457, 1475, 1468, 1486, 1489, 23512, 23517, 4541,
        4534, 4538, 4134, 4135, 40978, 40996, 41027, 40981, 40982, 40983, 40984,
        40701, 40670, 40685, 40900,  1111, 42732, 42733, 42734, 40498, 41161,
        41162, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41142, 41143,
        41144, 41145, 41146, 41147, 41150, 41156, 41157, 41158,  41159, 41138,
        54, 181, 188, 1461, 1494, 1464, 12, 23, 3, 1487, 40668, 1067, 1049,
        40975, 31]
    dataset_id = datasets[int(np.floor(index / num_seeds))]
    seed = index % num_seeds
    folder = os.path.expanduser('~/experiments/lccv/%d/%s/%d' % (dataset_id, algorithm, seed))
    os.makedirs(folder, exist_ok=True)
    run_experiment(dataset_id, algorithm, num_pipelines, seed, timeout, folder, prob_dp, prob_fp)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset_id is not None and args.algorithm is not None and args.seed is not None:
        run_experiment(args.dataset_id, args.algorithm, args.num_pipelines, args.seed, args.timeout, args.folder, args.prob_dp, args.prob_fp)
    elif args.experiment_idx is not None and args.num_seeds is not None and args.algorithm is not None:
        run_experiment_index_based(args.experiment_idx, args.num_seeds, args.algorithm, args.num_pipelines, args.timeout, args.prob_dp, args.prob_fp)
    else:
        raise ValueError('Wrong set of arguments provided. ')
