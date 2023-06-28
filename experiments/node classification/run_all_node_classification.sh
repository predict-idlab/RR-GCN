rm tuning_results.csv
python tune.py
rm evaluation_results.csv
python evaluate.py
rm evaluation_results_degree_cutting.csv
python evaluate_degree_cutting.py
rm evaluation_results_relation_cutting.csv 
python evaluate_relation_cutting.py 
pip install papermill
papermill tables.ipynb