# README.md

## Automating and Parallelizing Machine Learning Workflows with Snakemake

Managing machine learning workflows involves multiple repetitive and interdependent steps, from data preprocessing to model evaluation and reporting. This project leverages Snakemake, a workflow management system, to streamline, automate, and parallelize these tasks. With Snakemake’s rule-based framework, the workflow is modular, reproducible, and efficient, automatically handling dependencies and rerunning only required tasks when changes occur.

### **Purpose**
The goals of this project are:

- **Automation**: Streamlining the end-to-end machine learning pipeline, from preprocessing to reporting.
- **Parallelization**: Utilizing Snakemake to optimize resource usage and execution time.
- **Reproducibility**: Ensuring consistent results across different runs and environments.
- **Flexibility**: Supporting tasks like hyperparameter tuning and model comparison in a scalable and extensible framework.

### **Methodology**
The workflow is divided into the following steps:

1. **Preprocessing**:
   - The dataset (`diseaseDk_PCR_Hanze_WP2.xlsx`) is cleaned, with missing values imputed, and categorical features encoded.
   - The cleaned dataset is split into training and testing sets, saved as `train.csv` and `test.csv`.

2. **Model Training**:
   - Two models are trained:
     - **Random Forest**: A robust, tree-based model for classification.
     - **Ensemble Model**: Combines Random Forest and Gradient Boosting for improved predictions. 
     The scripts for training are trainens.py and trainRF.py and the models were saved 
     as "random_forest_model.pkl", "ensemble_model.pkl".

**3. Hyperparameter Tuning:**
    - The train.csv was fitted on the models and was tuned by hyperparameters.
    - The scripts for hyperparameter tuning  were tune_ensemble.py and tune_RandomForest.py and the outputs were  "RF_best_model.pkl" and "ensemble_best_model.pkl"


4. **Evaluation**:
   - By using the best models created in the previous step. The evaluation was done on the test dataset. The script was "evaluation.py" 
   - Metrics like accuracy, precision, recall, and F1-score are computed and saved as
   "random_forest_results.txt" and "ensemble_results.txt".

5. **Reporting**:
   - the output of the previous step("random_forest_results.txt" and "ensemble_results.txt") were used and "final_report.csv" was created as output. which shows all the metrics for both two models.
  and also generating several plot such as:

  "hyperparameter_plot.png" : used "ensemble_tuning_results.csv" to create this plot for ensemble model.

  "comparison_plot.png": which includes comparing the measured metric values for both models. It use generate_comparision_plot.py script as input. 



### **Requirements**

The following dependencies are required:

- **Environment Management**: Conda, mo_env
- **Workflow Management**: Snakemake
- **Python Libraries**:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - pyyaml
  - openpyxl
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - openpyxl
  - pyyaml
  - snakemake
```



### **Conclusion**

This project demonstrates the effective use of Snakemake to automate and parallelize machine learning workflows. Key benefits include:

- **Efficient Resource Utilization**: Parallel execution optimizes runtime.
- **Reproducibility**: Modular design ensures consistent results.
- **Scalability**: Workflow accommodates additional tasks and models.
