**Dask vs. Pandas for Gene Expression and Clinical Data Processing**

1. **Overview**

This project compares the performance of Dask and Pandas in processing gene expression and clinical datasets. It explores various data engineering challenges such as large-scale data processing, handling missing values, optimizing memory usage, and improving model performance using XGBoost.

2. **Purpose**

The primary objective of this study is to determine the efficiency and feasibility of using Dask over Pandas in genomic data processing. Key areas of focus include:
Speed and memory comparison for loading, preprocessing, and analyzing large datasets.
Efficient data transformations such as encoding, normalization, and feature extraction.
Parallelized execution of computations to handle high-dimensional data.
Model training performance on processed datasets.

3. **Methodology**
The following steps were performed to evaluate Dask and Pandas:
## Step 1: Data Loading
Gene expression data and clinical data were loaded using Pandas and Dask DataFrames.
Memory usage and execution times were recorded for comparison.
The module used for running the script is Data_loader.py
## Step 2: Data Preprocessing
Implemented Dask Delayed functions for:
Handling missing values in clinical data.
Filtering low-quality gene expression data.
Encoding categorical variables and normalizing numerical features.
Compared runtime and memory consumption between Pandas and Dask workflows.
The module used for running the script is initial_preprocessor.py

## Step 3: Exploratory Data Analysis (EDA)
Computed summary statistics (mean, median, standard deviation) for both datasets.
Identified top 10 most variable genes across patients.
Grouped patients based on clinical features to study gene expression trends.
Visualized TumorSubtype distributions using Dask and Pandas.
The module used for running the script is EDA.py

## Step 4: Data Splitting and Feature Engineering
Normalized gene expression data using z-score scaling.
Encoded TumorSubtype as a binary variable (1 for Squamous, 0 for others).
Splitted dataset into training and test sets using Dask.
The module used for running the script is further_processor.py

## Step 5: Model Training with XGBoost
Trained an XGBoost Classifier on the processed dataset.
Evaluated the model using key metrics:
Accuracy,Precision, Recall, F1-score, AUC-ROC
Measured runtime for training and inference using Dask and Pandas.
The module used for running the script is model.py

And the main script is dask_script.ipynb

###### ** **Execution**
To execute the entire workflow:
git clone https://github.com/Fatemeh1362/prog06.git


**Requirements**
The following dependencies are required:
Environment Management: Conda
Workflow Management: Dask
!pip install dask
!pip install xgboost[dask]
!pip uninstall xgboost dask-xgboost -y
!pip install --no-cache-dir xgboost[dask]
!pip install dask[complete]
!pip install xgboost==0.90
!pip install --upgrade dask-xgboost
!pip install numpy==1.24.2
!!pip install numpy==1.24.2
from dask.distributed import Client
!pip install dask[distributed] dask-xgboost scikit-learn numpy
import dask.dataframe as dd
from dask import delayed
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
!pip install dask
!pip install xlrd
import Data_loader 
from Data_loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
!pip install dask-ml



**Conclusion**:
This study demonstrates that Dask significantly improves efficiency in handling large-scale genomic data but comes with higher memory overhead. Pandas remains ideal for smaller datasets due to its simplicity and lower computational cost. A hybrid approach that utilizes Pandas for lightweight operations and Dask for heavy parallelized computations is the optimal workflow for bioinformatics data processing.
For further optimizations, Dask partition tuning, adaptive memory management, and efficient I/O handling should be explored in future work.