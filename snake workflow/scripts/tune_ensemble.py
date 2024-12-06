import argparse
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Tune Ensemble Model")
parser.add_argument("--input", required=True, help="Path to training data CSV file")
parser.add_argument("--results", required=True, help="Path to save tuning results CSV")
parser.add_argument("--model", required=True, help="Path to save the best model")
args = parser.parse_args()

# Load training data
train_data = pd.read_csv(args.input)

# Define features (X) and target (y)
X = train_data.drop(columns=["dk"])  # Replace "dk" with your target column
y = train_data["dk"]  # Replace "dk" with your target column

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Encode non-numeric features in X
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Define base classifiers
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Define the ensemble model
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='hard'  # Placeholder; tuning will adjust this
)

# Define the hyperparameter grid
param_grid = {
    "voting": ["hard", "soft"],  # Voting types
    "weights": [[1, 1], [2, 1], [1, 2], [3, 1], [1, 3]],  # Weight combinations for rf and gb
    "rf__n_estimators": [50, 100, 200],  # Hyperparameters for RandomForest
    "rf__max_depth": [None, 10, 20],
    "gb__n_estimators": [50, 100, 200],  # Hyperparameters for GradientBoosting
    "gb__learning_rate": [0.01, 0.1, 0.2],
}

# Perform grid search
grid_search = GridSearchCV(
    estimator=ensemble,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X, y)

# Save tuning results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df["model"] = "Ensemble"  # Add a column to identify the model
results_df.to_csv(args.results, index=False)

# Save the best model
joblib.dump(grid_search.best_estimator_, args.model)

# Print summary
print(f"Tuning complete. Best model saved to {args.model}")
print("Best Parameters:")
print(grid_search.best_params_)
