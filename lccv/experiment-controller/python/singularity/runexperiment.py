import sys
from evalutils import *
from lccv import lccv

if __name__ == '__main__':
    
    print("Starting python script")
    
    # read params
    folder = sys.argv[5]
    openmlid = int(sys.argv[1])
    algorithm = sys.argv[2]
    seed = int(sys.argv[3])
    timeout = int(sys.argv[4])
	
    
    # load data
    print("Reading dataset")
    X, y = get_dataset(openmlid)
    print("ready. Now running the algorithm")
    
    def mccv_adaptive(learner_inst, X, y, timeout, seed=0, r=None):
        return lccv(learner_inst, X, y, r = 1.0, eps = 0, timeout=timeout, base = 2, min_exp = np.log(int(np.floor(X.shape[0] * 0.9))) / np.log(2), MAX_EVALUATIONS = 10, seed=seed, enforce_all_anchor_evaluations=False, verbose=True)
    
    # creating learner sequence
    learners = [
        (sklearn.svm.LinearSVC, {}),
        (sklearn.tree.DecisionTreeClassifier, {}),
        (sklearn.tree.ExtraTreeClassifier, {}),
        (sklearn.linear_model.LogisticRegression, {}),
        (sklearn.linear_model.PassiveAggressiveClassifier, {}),
        (sklearn.linear_model.Perceptron, {}),
        (sklearn.linear_model.RidgeClassifier, {}),
        (sklearn.linear_model.SGDClassifier, {}),
        (sklearn.neural_network.MLPClassifier, {}),
        (sklearn.discriminant_analysis.LinearDiscriminantAnalysis, {}),
        (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis, {}),
        (sklearn.naive_bayes.BernoulliNB, {}),
        (sklearn.naive_bayes.MultinomialNB, {}),
        (sklearn.neighbors.KNeighborsClassifier, {}),
        (sklearn.ensemble.ExtraTreesClassifier, {}),
        (sklearn.ensemble.RandomForestClassifier, {}),
        (sklearn.ensemble.GradientBoostingClassifier, {})
    ]
    test_learners = [l[0] for l in learners]
    for _ in range(6):
        test_learners.extend(test_learners)
    random.seed(seed)
    test_learners = random.sample(test_learners, len(test_learners))
    test_learners = test_learners[:1000]
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
    
    result = evaluate_validators(validators, test_learners, X, y, timeout, repeats=100, epsilon=epsilon)[key]
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
