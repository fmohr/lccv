# Reproducing Results

## Creating The Result Figures and Tables of the Paper and Supplement
This section explains how the figures and tables in the paper and the supplement were computed based on these files of raw results:
* [lccv/publications/2022TPAMI/results/results_randomsearch.csv](https://raw.githubusercontent.com/fmohr/lccv/master/publications/2022TPAMI/results/results_randomsearch.csv)
* [lccv/publications/2022TPAMI/results/lccollection.csv.tar.gz](https://raw.githubusercontent.com/fmohr/lccv/master/publications/2022TPAMI/results/lccollection.csv.tar.gz)
* [lccv/publications/2022TPAMI/results/sensitivity.csv](https://raw.githubusercontent.com/fmohr/lccv/master/publications/2022TPAMI/results/sensitivity.csv)

The subsequent section below explains how these files were computed and how they can be reproduced.

Here is how each figure (except the schematic Fig. 1) in the main paper and in the appendix can be reproduced.
First prepare a suitable Python environment (tested for 3.9):
```bash
cd lccv/publications/2022TPAMI
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter-notebook
```

* Run the [analysis notebook for the evaluation of the validation techniques in Random Search](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/2%20-%20evaluation-randomsearch.ipynb) to produce
  * Fig. 2 in the main paper
  * Fig. 2, Fig. 3, Fig. 4, Fig. 5 in the appendix
  * Table 2 and Table 3 in the appendix

This notebook is built upon a file containing all results as computed in the above paragraph in a single file `results.csv`, which has been added to the repository for convenience.

* Run the [analysis notebook for the evaluation of the extrapolation techniques](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/4%20-%20evaluation-extrapolation.ipynb) to produce
  * Fig. 3 and Fig. 4 in the main paper

* Run the [dataset property notebook](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/0%20-%20dataset-properties.ipynb) to produce
  * Fig. 1 in the appendix
  * Table 1 in the appendix

* Run the [sensitivity analysis script](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/2%20-%20analyse_results_sensitivity.py)
  * to produce Fig. 6 in the appendix

  This script just expects the sensitivity results to be stored in a `sensitivity.csv` file (by default expected in `../results/`) and then will generate plots into the `plots/sensitivity` subfolder.

* Run the [convexity analysis notebook](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/5%20-%20convexity-analysis.ipynb)
  * to produce the figures with the learning curves in the appendix
  * to confirm that most learning curves exhibit a convex behavior


* Run the [pipeline sampling notebook](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/analysis/1%20-%20pipeline-sampling.ipynb) to generate the 10 sequences of 200 pipelines that were subject to the evaluation.

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

For ultimate reproduciblility, one could re-run all the experiments defined in the [experiment definition file](https://github.com/fmohr/lccv/blob/master/publications/2022TPAMI/experiment-controller/conf/experiments.conf) and gather the respective results from each `results.txt` into a single `results_randomsearch.csv` and locate it into the same folder as the analysis notebook.


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
