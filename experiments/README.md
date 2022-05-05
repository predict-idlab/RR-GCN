# Steps to reproduce paper results
## Node classification
1. `tune.py` to generate validation results for different hyperparameter setups. The resulting validation scores will be saved in `tuning_results.csv` (you need to delete the supplied version first if you want to run it yourself).
2. `evaluate.py` to generate test scores based on the optimal validation hyperparameters. This script requires a complete `tuning_results.csv` and saves test performance for both RR-GCN and RR-GCN-PPV in `evaluation_results.csv` (you need to delete the supplied version first if you want to run it yourself).
3. `evaluate_results_degree_cutting.py` to generate results with nodes of degree < 5 removed for AM, AIFB and BGS. This script requires a complete `tuning_results.csv` and saves test performance for both RR-GCN and RR-GCN-PPV in `evaluation_results_degree_cutting.csv` (you need to delete the supplied version first if you want to run it yourself).
4. `evaluate_results_relation_cutting.py` to generate results with nodes of degree < 5 and unimportant relations (as extracted from a trained R-GCN) removed for AM. This script requires a complete `tuning_results.csv` and saves test performance for both RR-GCN and RR-GCN-PPV in `evaluation_results_relation_cutting.csv` (you need to delete the supplied version first if you want to run it yourself).
5. `tables.ipynb` aggregates all previously generated csv-files into the results used in the paper.

## Link prediction
1. `link_prediction.ipynb` prints the results reported in the paper in the last cell.