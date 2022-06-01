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
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--algorithm', type=str, choices=['cv', 'lccv', 'lccv-flex', 'wilcoxon', 'sh'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max_memory', type=int, default=14) # allowed memory in GB
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--final_repeats', type=int, default=100)
    parser.add_argument('--num_pipelines', type=int, default=1000)
    parser.add_argument('--folder', type=str, default=os.path.expanduser(default_path))
    parser.add_argument('--prob_dp', type=float, default=0.5)
    parser.add_argument('--prob_fp', type=float, default=0.5)
    return parser.parse_args()


def run_experiment(openmlid: int, train_size: float, algorithm: str, num_pipelines: int, seed: int, timeout: int, folder: str, prob_dp: float, prob_fp: float, final_repeats: int, max_memory: int):
    # TODO: built in check whether file already exists, in that case we can skipp
    
    if train_size < 0.05 or train_size > 0.95:
        raise ValueError(f"train_size must be between 0.05 and 0.95 but is {train_size}.")
    
    # CPU
    print("CPU Settings:")
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        print(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")
        
    # memory limits
    memory_limit = max_memory * 1024
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
    Maximum Training Portion: {train_size}
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
    memory_limit = max_memory * 1024
    exp_logger.info("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    

    
    # load data
    binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
    exp_logger.info(f"Reading dataset. Will be binarized sparsely: {binarize_sparse}")
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
    
    print(f"Checking that no pipeline contains a copy or warm-starting parameter!")
    for i in tqdm(range(num_pipelines)):
        pl = sampler.sample(do_build=True)
        pl_str = str(pl)
        if "copy" in pl_str:
            raise Exception(f"{i+1}-th pipeline has a copy command! Full pipeline description: {pl_str}")
        if "warm_start" in pl_str:
            raise Exception(f"{i+1}-th pipeline has a warm-start command! Full pipeline description: {pl_str}")
    
    test_learners = [sampler.sample(do_build=False) for i in range(num_pipelines)]    
    
    exp_logger.info(f"Evaluating portfolio of {len(test_learners)} learners.")
    
    
    if algorithm == "sh":
        max_train_size = train_size
        if train_size == 0.8:
            repeats = 5
        elif train_size == 0.9:
            repeats = 10
        else:
            raise ValueError(f"train_size for sh must be 0.8 or 0.9 since the number of repetitions is not well-defined otherwise.")
            
        selector = SH(X, y, binarize_sparse, timeout, max_train_budget = max_train_size, seed=seed, repeats = repeats)
    else:
        selector = VerticalEvaluator(X, y, binarize_sparse, algorithm, train_size, timeout, epsilon = 0.01, seed=seed)
    
    # run selector
    time_start = time.time()
    model = selector.select_model(test_learners)
    runtime = time.time() - time_start
    
    print("\n-------------------\n\n")
    
    if model is not None:
        
        # compute validation performance of selection
        error_rates = selector.mccv(model, target_size=train_size, timeout=None, seed=seed, repeats = final_repeats)
        error_rates = [np.round(r, 4) for r in error_rates if not np.isnan(r)]
        error_rate = np.mean(error_rates)
        model_name = str(model).replace("\n", " ")
        exp_logger.info(f"""Run completed. Here are the details:
            Model: {model}
            Error Rate: {error_rate}
            Runtime: {runtime}
            {len(error_rates)}/{final_repeats} valid results in final evaluation: {np.array(error_rates)}""")
    else:
        exp_logger.info("No model was chosen. Assigning maximum error rate")
        error_rate = 1
        model_name = "None"
        
    # write result
    output = (model_name, np.nanmean(error_rate), error_rates, runtime)
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
        run_experiment(args.dataset_id, args.train_size, args.algorithm, args.num_pipelines, args.seed, args.timeout, args.folder, args.prob_dp, args.prob_fp, args.final_repeats, args.max_memory)
    elif args.experiment_idx is not None and args.num_seeds is not None and args.algorithm is not None:
        run_experiment_index_based(args.experiment_idx, args.num_seeds, args.algorithm, args.num_pipelines, args.timeout, args.prob_dp, args.prob_fp)
    else:
        raise ValueError('Wrong set of arguments provided. Specify either\n\t--dataset_id=.. --train_size=.. --algorithm=.. --seed=..  or\n\t--experiment_idx=.. --algorithm=.. --num_seeds=.. ')
