import sys
from evalutils import *
from lccv import lccv
from commons import *

if __name__ == '__main__':
    
    print("Starting python script")
    
    # read params
    folder = sys.argv[5]
    openmlid = int(sys.argv[1])
    algorithm = sys.argv[2]
    seed = int(sys.argv[3])
    timeout = int(sys.argv[4])
    num_pipelines = 10
    print("Running experiment under folloiwing conditions:")
    print("\tOpenML id:", openmlid)
    print("\tAlgorithm:", algorithm)
    print("\tSeed:", seed)
    print("\ttimeout (per single evaluation):", timeout)
    print("\tNum Pipelines:", num_pipelines)
	
    
    # load data
    print("Reading dataset")
    X, y = get_dataset(openmlid)
    print("ready. Now running the algorithm")
    
    def mccv_adaptive(learner_inst, X, y, timeout, seed=0, r=None):
        return lccv(learner_inst, X, y, r = 1.0, eps = 0, timeout=timeout, base = 2, min_exp = np.log(int(np.floor(X.shape[0] * 0.9))) / np.log(2), MAX_EVALUATIONS = 10, seed=seed, enforce_all_anchor_evaluations=False, verbose=True)
    
    # creating learner sequence
    sampler = PipelineSampler("searchspace.json", X, y, dp_proba = .5, fp_proba = .5)
    test_learners = [sampler.sample() for i in range(num_pipelines)]
    print("Evaluating portfolio of " + str(len(test_learners)) + " learners.")
    
    # run lccv
    epsilon = 0.0
    if "10cv" in algorithm:
        if "adaptive" in algorithm:
            validators = validators = [(mccv_adaptive, lambda r: r[0])]
            key = "mccv_adaptive"
        else:
            validators = validators = [(mccv, lambda r: r[0])]
            key = "mccv"
    if "lccv" in algorithm:
        if "adaptive" in algorithm:
            validators = validators = [(lccv, lambda r: r[0])]
            key = "lccv"
    
    result = evaluate_validators(validators, test_learners, X, y, timeout, seed=seed, repeats=100, epsilon=epsilon)[key]
    model = result[0]
    runtime = result[1]
    error_rate = np.round(result[2][0], 4)
    print("Model:", model)
    print("Error Rate:", error_rate)
    print("Runtime:",runtime)
    print("Results in final evaluation:", np.round(result[2][1], 4))
    
    # write result
    f = open(folder + "/results.txt", "w")
    f.write(str(model) + " " + str(error_rate) + " " + str(runtime))
    f.close()
    print("Experiment ready. Results written to", folder + "/results.txt")
