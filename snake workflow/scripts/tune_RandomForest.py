import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import argparse
import joblib

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Tune Random Forest")
parser.add_argument("--train", required=True, help="Path to training data CSV file")  # Path to the training dataset
parser.add_argument("--results", required=True, help="Path to save tuning results CSV")  # Path to save the hyperparameter tuning results
parser.add_argument("--model", required=True, help="Path to save the best model")  # Path to save the best trained model
args = parser.parse_args()

# Load training data
train_data = pd.read_csv(args.train)  # Read the training dataset from the provided path

# Define features (X) and target (y)
X = train_data.drop(columns=["dk"])  # Drop the target column to define the feature set
y = train_data["dk"]  # Extract the target column

# Encode categorical features in X
for col in X.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()  # Initialize a label encoder for categorical data
    X[col] = encoder.fit_transform(X[col])  # Encode categorical features into numerical values

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],  # Number of trees in the forest
    "max_depth": [None, 10, 20],  # Maximum depth of the tree (None means no limit)
    "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
}

# Perform grid search
rf = RandomForestClassifier(random_state=42)  # Initialize a Random Forest classifier with a fixed random seed
grid_search = GridSearchCV(
    estimator=rf,  # Model to be optimized
    param_grid=param_grid,  # Grid of hyperparameters to test
    cv=5,  # Number of cross-validation folds
    scoring="accuracy",  # Metric to evaluate model performance
    error_score="raise",  # Raise an error if a hyperparameter configuration fails
)
grid_search.fit(X, y)  # Train the model with the grid search process

# Save tuning results
results_df = pd.DataFrame(grid_search.cv_results_)  # Convert grid search results to a DataFrame
results_df.to_csv(args.results, index=False)  # Save the results to the specified path

# Save the best model
joblib.dump(grid_search.best_estimator_, args.model)  # Save the best model to a file

# Print confirmation message
print(f"Best model saved to {args.model}")  # Notify the user that the best model has been saved
