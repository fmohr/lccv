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
    
    num_seeds = 10
	
    
    # load data
    print("Reading dataset")
    X, y = get_dataset(openmlid)
    labels = list(np.unique(y))
    is_binary = len(labels) == 2
    minority_class = labels[np.argmin([np.count_nonzero(y == label) for label in labels])]
    print(f"minority_class is {minority_class}")
    print("ready. Now building the learning curve")
    
    min_exp = 4
    max_size = X.shape[0] * 0.9
    max_exp = np.log(max_size) / np.log(2)
    max_exp_int = (int(np.floor(max_exp)) - min_exp) * 2
    anchors = [int(np.round(2**(min_exp + 0.5 * exp))) for exp in range(0, max_exp_int + 1)]
   
    if anchors[-1] != int(max_size):
        anchors.append(int(max_size))
    
    
    print("Anchors:", anchors)
    
    def get_class( kls ):
        parts = kls.split('.')
        module = ".".join(parts[:-1])
        m = __import__( module )
        for comp in parts[1:]:
            m = getattr(m, comp)            
        return m

    
    def get_truth_and_predictions(learner_inst, X, y, num_examples, seed=0, timeout = None, verbose=False):
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

        start_time = time.time()
        if verbose:
            print("Training " + str(learner_inst) + " on data of shape " + str(X_train.shape) + " using seed " + str(seed))
        if deadline is None:
            learner_inst.fit(X_train, y_train)
        else:
            func_timeout(deadline - time.time(), learner_inst.fit, (X_train, y_train))
        train_time = time.time() - start_time

        if verbose:
            print("Training ready. Obtaining predictions for " + str(X_test.shape[0]) + " instances.")
        
        # compute predictions on train data
        start_time = time.time()
        y_hat_train = learner_inst.predict(X_train)
        predict_time_train = time.time() - start_time
        start_time = time.time()
        try:
            y_prob_train = learner_inst.predict_proba(X_train)
        except:
            y_prob_train = None
        predict_proba_time_train = time.time() - start_time
        
        # compute predictions on test data
        start_time = time.time()
        y_hat_test = learner_inst.predict(X_test)
        predict_time_test = time.time() - start_time
        start_time = time.time()
        try:
            y_prob_test = learner_inst.predict_proba(X_test)
        except:
            y_prob_test = None
        predict_proba_time_test = time.time() - start_time
        return y_train, y_test, y_hat_train, y_prob_train, y_hat_test, y_prob_test, train_time, predict_time_train, predict_proba_time_train, predict_time_test, predict_proba_time_test

    #
    out = []
    #constructor = globals()[algorithm]
    learner_inst = get_class(algorithm)()
    for i in range(num_seeds):
        seed = num_seeds * seed_index + i
        try:
            for anchor in anchors:
                tic = time.time()
                y_train, y_test, y_hat_train, y_prob_train, y_hat_test, y_prob_test, train_time, predict_time_train, predict_proba_time_train, predict_time_test, predict_proba_time_test = get_truth_and_predictions(learner_inst, X, y, anchor, seed, verbose=True)
                
                # compute metrics
                info = {
                    "traintime": np.round(train_time, 4), 
                    "predicttime_train": np.round(predict_time_train, 4),
                    "predicttimeproba_train": np.round(predict_proba_time_train, 4),
                    "predicttime_test": np.round(predict_time_test, 4),
                    "predicttimeproba_test": np.round(predict_proba_time_test, 4),
                    "seed": seed,
                    "size_train": anchor,
                    "size_test": len(y_test)
                }
                
                for suffix, y_true, y_hat, y_prob in [("_train", y_train, y_hat_train, y_prob_train), ("_test", y_test, y_hat_test, y_prob_test)]:
                    
                    # accuracy
                    info["accuracy" + suffix] = np.round(sklearn.metrics.accuracy_score(y_true, y_hat), 4)
                    
                    # log-loss
                    if y_prob is not None:
                        try:
                            info["log_loss" + suffix] = np.round(sklearn.metrics.log_loss(y_true, y_prob, labels=labels), 4)
                        except:
                            pass
                    
                    # binary classification metrics
                    if is_binary:
                        if y_prob is not None:
                            if y_prob.shape[1] == 2:
                                info["auc" + suffix] = np.round(sklearn.metrics.roc_auc_score(y_true, y_prob[:, 1], labels=np.unique(y)), 4)
                        y_true_pos = y_true == minority_class
                        y_pred_pos = y_hat == minority_class
                        info["tp" + suffix] = np.count_nonzero(y_true_pos & y_pred_pos)
                        info["fp" + suffix] = np.count_nonzero(~y_true_pos & y_pred_pos)
                        info["fn" + suffix] = np.count_nonzero(y_true_pos & ~y_pred_pos)
                        info["tn" + suffix] = np.count_nonzero(~y_true_pos & ~y_pred_pos)

                        if info["tp" + suffix] + info["fp" + suffix] + info["tn" + suffix] + info["fn" + suffix] != info["size" + suffix]:
                            raise Exception("Count is not correct!")
                
                out.append(info)
        except Exception:
            print("AN ERROR OCCURED!")
        print("Progress:", str(np.round(100 * i / num_seeds, 2)) + "%")
    with open(file, 'w') as outfile:
        json.dump(out, outfile)