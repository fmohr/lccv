import sys
import resource
from evalutils import *
from lccv import lccv
from commons import *
import logging
import json

if __name__ == '__main__':
    
    print("Starting python script")
    
    # read params
    folder = sys.argv[5]
    openmlid = int(sys.argv[1])
    algorithm = sys.argv[2]
    seed = int(sys.argv[3])
    timeout = int(sys.argv[4])
    num_pipelines = 1000
    print("Running experiment under folloiwing conditions:")
    print("\tOpenML id:", openmlid)
    print("\tAlgorithm:", algorithm)
    print("\tSeed:", seed)
    print("\ttimeout (per single evaluation):", timeout)
    print("\tNum Pipelines:", num_pipelines)
    
    # CPU
    print("CPU Settings:")
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        print(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")
        
    # memory limits
    memory_limit = 2 * 1024
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
    
    # load data
    print("Reading dataset")
    X, y = get_dataset(openmlid)
    print("ready. Now running the algorithm")
    
    def mccv_adaptive(learner_inst, X, y, timeout, seed=0, r=None):
        return lccv(learner_inst, X, y, r = 1.0, eps = 0, timeout=timeout, base = 2, min_exp = np.log(int(np.floor(X.shape[0] * 0.9))) / np.log(2), MAX_EVALUATIONS = 10, seed=seed, enforce_all_anchor_evaluations=False, verbose=True)
    
    # creating learner sequence
    sampler = PipelineSampler("searchspace.json", X, y, seed, dp_proba = .5, fp_proba = .5)
    test_learners = [sampler.sample() for i in range(num_pipelines)]
    print("Evaluating portfolio of " + str(len(test_learners)) + " learners.")
    
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
    else:
        raise Exception(f"Unsupported validation algorithm {algorithm}")
    result = evaluate_validators(validators, test_learners, X, y, timeout, seed=seed, repeats=100, epsilon=epsilon)[key]
    model = result[0]
    runtime = result[1]
    if model is not None:
        error_rate = np.round(result[2][0], 4)
        model_name = str(model).replace("\n", " ")
        print("Model:", model)
        print("Error Rate:", error_rate)
        print("Runtime:",runtime)
        print("Results in final evaluation:", np.round(result[2][1], 4))
    else:
        print("No model was chosen. Assigning maximum error rate")
        error_rate = 1
        model_name = "None"
    
    # write result
    output = (model_name, error_rate, runtime)
    with open(folder + "/results.txt", "w") as outfile: 
        json.dump(output, outfile)
    print("Experiment ready. Results written to", folder + "/results.txt")
