import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Argument parser
parser = argparse.ArgumentParser(description="Train Random Forest model")
parser.add_argument("--train", required=True, help="Path to the training data CSV")
parser.add_argument("--output", required=True, help="Path to save the trained model")
args = parser.parse_args()

# Load training data
print(f"Loading training data from {args.train}")
train_data = pd.read_csv(args.train)

# Define features and target
target_column = "dk"
features = [col for col in train_data.columns if col != target_column]
X_train = train_data[features]
y_train = train_data[target_column]

# Handle non-numeric columns
print("Preprocessing features...")
categorical_cols = X_train.select_dtypes(include=["object"]).columns
numeric_cols = X_train.select_dtypes(include=["number"]).columns

# Preprocessing for numerical and categorical columns
numeric_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Create a pipeline with preprocessing and model
rf = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))])

# Train the model
print("Training Random Forest model...")
rf.fit(X_train, y_train)

# Save the trained model
print(f"Saving Random Forest model to {args.output}")
with open(args.output, "wb") as f:
    pickle.dump(rf, f)

print("Random Forest training completed successfully.")
