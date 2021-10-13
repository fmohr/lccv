import sys
from evalutils import *
from lccv import lccv
import sklearn.tree
import json

if __name__ == '__main__':
    
    print("Starting python script")
    
    # read params
    openmlid = int(sys.argv[1])
    algorithm = sys.argv[2]
    seed_index = int(sys.argv[3])
    file = sys.argv[4]
	
    
    # load data
    print("Reading dataset")
    X, y = get_dataset(openmlid)
    print("ready. Now building the learning curve")
    
    max_size = X.shape[0] * 0.9
    max_exp = np.log(max_size) / np.log(2)
    max_exp_int = int(np.floor(max_exp))
    anchors = [2**exp for exp in range(2, max_exp_int + 1)]
    if max_exp_int < max_exp:
        anchors.append(int(2**max_exp))
    
    
    print("Anchors:", anchors)
    
    def get_class( kls ):
        parts = kls.split('.')
        module = ".".join(parts[:-1])
        m = __import__( module )
        for comp in parts[1:]:
            m = getattr(m, comp)            
        return m

    
    def evaluate(learner_inst, X, y, num_examples, seed=0, timeout = None, verbose=False):
        deadline = None if timeout is None else time.time() + timeout
        random.seed(seed)
        n = X.shape[0]
        indices_train = random.sample(range(n), num_examples)
        mask_train = np.zeros(n)
        mask_train[indices_train] = 1
        mask_train = mask_train.astype(bool)
        mask_test = (1 - mask_train).astype(bool)
        X_train = X[mask_train]
        y_train = y[mask_train]
        X_test = X[mask_test][:10000]
        y_test = y[mask_test][:10000]

        if verbose:
            print("Training " + str(learner_inst) + " on data of shape " + str(X_train.shape) + " using seed " + str(seed))
        if deadline is None:
            learner_inst.fit(X_train, y_train)
        else:
            func_timeout(deadline - time.time(), learner_inst.fit, (X_train, y_train))


        y_hat = learner_inst.predict(X_test)
        error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
        if verbose:
            print("Training ready. Obtaining predictions for " + str(X_test.shape[0]) + " instances. Error rate of model on " + str(len(y_hat)) + " instances is " + str(error_rate))
        return error_rate

    #
    out = []
    #constructor = globals()[algorithm]
    learner_inst = get_class(algorithm)()
    for i in range(50):
        seed = 50 * seed_index + i
        try:
            for anchor in anchors:
                tic = time.time()
                er = evaluate(learner_inst, X, y, anchor, seed, verbose=True)
                toc = time.time()
                runtime = toc - tic
                out.append({"errorrate": er, "runtime": runtime, "seed": seed, "trainsize": anchor})
        except Exception:
            print("AN ERROR OCCURED!")
        print("Progress:", str(np.round(100 * i / 50, 2)) + "%")
    with open(file, 'w') as outfile:
        json.dump(out, outfile)