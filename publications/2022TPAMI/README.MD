# Reproducing Results

## Comparison of LCCV vs other techniques in Random Search

The essential experiment program code can be found in (lccv/publications/2022TPAMI/experiment-controller/python/singularity/).

To replicate a specific experiment, you can proceed as follows (tested only under Linux):
```
cd lccv/publications/2022TPAMI/experiment-controller/python/singularity
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python runexperiment.py --dataset_id=61 --train_size=0.8 --seed=0 --algorithm=lccv-flex --folder=. --num_pipelines=20
```
Here,
- `dataset_id` is the openml dataset id
- `algorithm` is one of: `cv`, `lccv`, `lccv-flex`, `wilcoxon`, `sh` (in the paper, lccv was evaluated through `lccv-flex`)
- `train_size` is the value used for training a model in any CV, typically 0.8 or 0.9

The results for this specific dataset/algorithm/seed combination will be stored in a file called `results.txt` in the `folder` you provided. 
At the time of running the experiments, [py-experimenter](https://github.com/tornede/py_experimenter) had not yet been released so that experiment management was conducted through the respective Java API. However, the main experiment code itself is in Python.
