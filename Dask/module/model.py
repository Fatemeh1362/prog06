import pandas as pd
import dask.array as da
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


class XGBoostTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialize the trainer with training and testing datasets.
        Automatically preprocesses data.
        """
        print("\n Initializing XGBoostTrainer...")

        # Convert Dask Arrays to NumPy if necessary
        if isinstance(X_train, da.Array):
            X_train = X_train.compute()
        if isinstance(y_train, da.Array):
            y_train = y_train.compute()
        if isinstance(X_test, da.Array):
            X_test = X_test.compute()
        if isinstance(y_test, da.Array):
            y_test = y_test.compute()

        # Ensure targets are 1D
        self.y_train = y_train.ravel()
        self.y_test = y_test.ravel()

        # Preprocess the features
        self.X_train = self.preprocess_data(X_train)
        self.X_test = self.preprocess_data(X_test)

        # Initialize the model
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

        print("\n XGBoostTrainer Initialized Successfully!")

    def preprocess_data(self, X):
        """
        Convert categorical variables to numeric and ensure all columns are float32.
        """
        print("\n Checking for non-numeric columns in X...")

        if isinstance(X, pd.DataFrame):
            # Identify categorical columns
            categorical_columns = X.select_dtypes(include=["string[pyarrow]", "object"]).columns

            if len(categorical_columns) > 0:
                print(f" Found categorical columns: {list(categorical_columns)}")

                # Apply Label Encoding for categorical columns
                le = LabelEncoder()
                for col in categorical_columns:
                    X[col] = le.fit_transform(X[col].astype(str))

        # Convert all columns to float32
        X = X.astype("float32")

        print(f" Data Preprocessing Completed. Shape: {X.shape}")
        return X

    def train_model(self):
        """
        Train the XGBoost model using the preprocessed training data.
        """
        print("\n Training XGBoost Model...")

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        print(f" Training Completed in {training_time:.4f} seconds")
        return training_time

    def evaluate_model(self):
        """
        Evaluate the trained model on the test dataset.
        """
        print("\n Evaluating Model...")

        start_time = time.time()
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)
        inference_time = time.time() - start_time

        print(f" Inference Completed in {inference_time:.4f} seconds")

        # Compute evaluation metrics
        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred, average="weighted", zero_division=1),
            "Recall": recall_score(self.y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(self.y_test, y_pred, average="weighted"),
        }

        # Handle AUC-ROC for binary classification
        try:
            metrics["AUC-ROC"] = roc_auc_score(self.y_test, y_proba[:, 1])
        except ValueError as e:
            print(f" Warning: AUC-ROC could not be calculated. Reason: {e}")
            metrics["AUC-ROC"] = None

        metrics["Inference Time (s)"] = inference_time
        return metrics

