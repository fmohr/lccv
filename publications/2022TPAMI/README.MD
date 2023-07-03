# Reproducing Results

## Creating The Result Figures of the Paper and Supplement
This section explains how the figures and tables in the paper and the supplement were computed based on these files of raw results:
* [lccv/publications/2022TPAMI/results/results_randomsearch.csv](https://raw.githubusercontent.com/fmohr/lccv/master/publications/2022TPAMI/results/results_randomsearch.csv)
* [lccv/publications/2022TPAMI/results/lccollection.csv.tar.gz](https://raw.githubusercontent.com/fmohr/lccv/master/publications/2022TPAMI/results/lccollection.csv.tar.gz)
* [lccv/publications/2022TPAMI/results/sensitivity.csv](https://raw.githubusercontent.com/fmohr/lccv/master/publications/2022TPAMI/results/sensitivity.csv)

The following section below explains how these files were computed and how they can be reproduced.

For the figures in the main paper and the comparative plots in the appendix, this is done via the [analysis notebook](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/2%20-%20evaluation-randomsearch.ipynb).
This notebook is built upon a file containing all results as computed in the above paragraph in a single file `results.csv`, which has been added to the repository for convenience.
For ultimate reproduciblility, one could re-run all the experiments defined in the [experiment definition file](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/experiment-controller/conf/experiments.conf) and gather the respective results from each `results.txt` into a single `results.csv` and locate it into the same folder as the analysis notebook.

For the sensitivity analysis in the appendix, it is done via the [sensitivity analysis script](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/analyse_results_sensitivity.py), which just expects the sensitivity results to be stored in a `sensitivity.csv` file (by default expected in `../results/`) and then will generate plots into the `plots/sensitivity` subfolder.


## Gathering Raw Results

### Computation of results of LCCV and other techniques when being used in Random Search
The essential experiment program code can be found in [lccv/publications/2022TPAMI/experiment-controller/python/singularity/runexperiment.py](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/experiment-controller/python/singularity/runexperiment.py).

To replicate a specific experiment, you can proceed as follows (tested only under Linux):
```bash
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

Results for the sensitivity analysis can be gathered in pretty much the same way but using the [lccv/publications/2022TPAMI/experiment-controller/python/singularity/run_sensitivity.py](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/experiment-controller/python/singularity/run_sensitivity.py) script.

Usage example:
``` bash
python run_sensitivity.py --experiment_idx=2 --dataset_id=61 --num_pipelines=20
```

### Collecting Learning Curves
The learning curves for datasets and learners can be collected through this script.
```bash
python computelc.py <dataset_id> <sklearn_class_name> <seed> <outfile>
```
The results of all such runs were condensed into a CSV file and compressed into [lccv/publications/2022TPAMI/results/lccollection.csv.tar.gz](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/results/lccollection.csv.tar.gz), which is used for the second part of the evaluation and the analysis of the curves on convexity (for example the plots in the supplement).
