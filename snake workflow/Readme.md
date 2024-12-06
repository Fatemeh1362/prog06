**Automating and Parallelizing Machine Learning Workflows with Snakemake**
Managing machine learning workflows involves multiple repetitive and interdependent steps, from data preprocessing to model evaluation and reporting. This project leverages Snakemake, a workflow management system, to streamline, automate, and parallelize these tasks. With Snakemake’s rule-based framework, the workflow is modular, reproducible, and efficient, automatically handling dependencies and rerunning only required tasks when changes occur.

**Purpose**
The goals of this project are:

Automation: Streamlining the end-to-end machine learning pipeline, from preprocessing to reporting.
Parallelization: Utilizing Snakemake to optimize resource usage and execution time.
Reproducibility: Ensuring consistent results across different runs and environments.
Flexibility: Supporting tasks like hyperparameter tuning and model comparison in a scalable and extensible framework.
Workflow Overview
The workflow consists of the following steps:

**Preprocessing:**

The dataset (diseaseDk_PCR_Hanze_WP2.xlsx) is cleaned, missing values are imputed, and categorical features are encoded.
The dataset is split into training (train_data.csv) and testing (test_data.csv) datasets.
Script Used: preprocessor.py.

**Model Training and Hyperparameter Tuning:**

###### Random Forest Model:
Training: Generates the base Random Forest model (random_forest_model.pkl).
Tuning: Optimizes hyperparameters and outputs the best model (RF_best_model.pkl) and results (RF_tuning_results.csv).
Scripts Used: trainRF.py and tune_RandomForest.py.

###### Ensemble Model:
Training: Generates the Ensemble model (ensemble_model.pkl).
Tuning: Optimizes hyperparameters and outputs the best model (ensemble_best_model.pkl) and results (ensemble_tuning_results.csv).
Scripts Used: trainens.py and tune_ensemble.py.


**Model Evaluation:**

The best models are evaluated on the test dataset (test_data.csv).
Metrics such as accuracy, precision, recall, and F1-score are saved as random_forest_results.txt and ensemble_results.txt.
Script Used: evaluation.py.

**Reporting:**

A final report (final_report.csv) is generated comparing the performance of both models.
Script Used: Create_report.py.

**Visualization:**

###### Hyperparameter Tuning Plot:
A plot visualizing tuning results for the Ensemble model (hyperparameter_plot.png).
Script Used: plot_hyperparameter_results.py.

###### Comparison Plot:
A plot comparing evaluation metrics of both models (comparison_plot.png).
Script Used: generate_comparison_plot.py.

**Snakemake Rules**
The Snakemake workflow is structured as follows:

preprocess_data: Prepares the dataset.
tune_random_forest and tune_ensemble: Optimize hyperparameters for the respective models.
evaluate_random_forest and evaluate_ensemble: Evaluate the performance of the tuned models.
create_report: Consolidates evaluation metrics into a report.
plot_hyperparameter_results and generate_comparison_plot: Creates visualizations for hyperparameter tuning and model comparison.


**Requirements**
The following dependencies are required:
Environment Management: Conda
Workflow Management: Snakemake
Python Libraries:
scikit-learn
pandas
numpy
matplotlib
pyyaml
openpyxl


**Execution**
To execute the entire workflow:
snakemake -s rules/main.smk --cores <4>


To visualize the workflow DAG:
snakemake -s rules/main.smk --dag | dot -Tpng > workflow_dag.png

**Conclusion**
This project demonstrates the effective use of Snakemake to automate and parallelize machine learning workflows. Key benefits include:

Efficient Resource Utilization: Parallel execution optimizes runtime.
Reproducibility: Modular design ensures consistent results.
