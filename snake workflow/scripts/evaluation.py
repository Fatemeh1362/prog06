import argparse
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate trained model")
parser.add_argument("--test", required=True, help="Path to the testing data CSV")
parser.add_argument("--model", required=True, help="Path to the trained model file")
parser.add_argument("--output", required=True, help="Path to save evaluation results")
args = parser.parse_args()

# Load test data
print(f"Loading testing data from {args.test}")
test_data = pd.read_csv(args.test)

# Define features and target
target_column = "dk"
features = [col for col in test_data.columns if col != target_column]
X_test = test_data[features]
y_test = test_data[target_column]

# Load the trained model
print(f"Loading model from {args.model}")
with open(args.model, "rb") as f:
    model = pickle.load(f)

# Make predictions
print("Making predictions...")
y_pred_continuous = model.predict(X_test)  # Continuous predictions
y_pred_binary = (y_pred_continuous > 0.5).astype(int)  # Convert to binary (threshold = 0.5)

# Evaluate the model
print("Evaluating the model...")
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, average="binary")
recall = recall_score(y_test, y_pred_binary, average="binary")
f1 = f1_score(y_test, y_pred_binary, average="binary")
mse = mean_squared_error(y_test, y_pred_continuous)
r2 = r2_score(y_test, y_pred_continuous)

# Save evaluation results
print(f"Saving evaluation results to {args.output}")
with open(args.output, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

print("Evaluation completed successfully.")
