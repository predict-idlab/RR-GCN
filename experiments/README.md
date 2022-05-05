# Steps to reproduce paper results
## Prerequisites
For these scripts to work, the RR-GCN package needs to be installed (`pip install -e .` in the parent directory) and this directory needs to be the working directory. Note that node classification results will only exactly match those reported in the paper as CatBoost on CPU and GPU produce different results.

1. Make sure `git-lfs` is installed and working on your machine
2. Edit [`dataset_root.py`](dataset_root.py) and set `root` equal to a path where datasets can be downloaded

## Node classification
1. [`tune.py`](tune.py) to generate validation results for different hyperparameter setups. The resulting validation scores will be saved in [`tuning_results.csv`](tuning_results.csv) (you need to delete the supplied version first if you want to run it yourself).
2. [`evaluate.py`](evaluate.py) to generate test scores based on the optimal validation hyperparameters. This script requires a complete [`tuning_results.csv`](tuning_results.csv) and saves test performance for both RR-GCN and RR-GCN-PPV in [`evaluation_results.csv`](evaluation_results.csv) (you need to delete the supplied version first if you want to run it yourself).
3. [`evaluate_results_degree_cutting.py`](evaluate_results_degree_cutting.py) to generate results with nodes of degree < 5 removed for AM, AIFB and BGS. This script requires a complete [`tuning_results.csv`](tuning_results.csv) and saves test performance for both RR-GCN and RR-GCN-PPV in [`evaluation_results_degree_cutting.csv`](evaluation_results_degree_cutting.csv) (you need to delete the supplied version first if you want to run it yourself).
4. [`evaluate_results_relation_cutting.py`](evaluate_results_relation_cutting.py) to generate results with nodes of degree < 5 and unimportant relations (as extracted from a trained R-GCN) removed for AM. This script requires a complete [`tuning_results.csv`](tuning_results.csv) and saves test performance for both RR-GCN and RR-GCN-PPV in [`evaluation_results_relation_cutting.csv`](evaluation_results_relation_cutting.csv) (you need to delete the supplied version first if you want to run it yourself).
5. [`tables.ipynb`](tables.ipynb) aggregates all previously generated csv-files into the results used in the paper.

## Link prediction
1. [`link_prediction.ipynb`](link_prediction.ipynb) prints the results reported in the paper in the last cell.