import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Plot hyperparameter tuning results.")
parser.add_argument("--input", required=True, help="Path to the input CSV file.")
parser.add_argument("--output", required=True, help="Path to save the plot.")
args = parser.parse_args()

# Load the hyperparameter tuning results
results = pd.read_csv(args.input)

# Identify hyperparameter columns dynamically based on the column names in the CSV
rf_hyperparameter_columns = [col for col in results.columns if col.startswith("param_rf__")]
ensemble_hyperparameter_columns = [col for col in results.columns if col.startswith("param_gb__") or col.startswith("param_weights")]

# Dynamically create the 'hyperparameter_set' column for each model type
if rf_hyperparameter_columns:
    results["RF_hyperparameter_set"] = results[rf_hyperparameter_columns].astype(str).agg("_".join, axis=1)

if ensemble_hyperparameter_columns:
    results["Ensemble_hyperparameter_set"] = results[ensemble_hyperparameter_columns].astype(str).agg("_".join, axis=1)

# Check if required columns are in the input file
required_columns = ["model", "mean_test_score"]
for col in required_columns:
    if col not in results.columns:
        raise ValueError(f"Missing required column: {col}")

# Create a combined plot
plt.figure(figsize=(14, 8))

# Plot Random Forest results
if rf_hyperparameter_columns:
    rf_subset = results[results["model"] == "random_forest"]
    if not rf_subset.empty:
        plt.plot(rf_subset["RF_hyperparameter_set"], rf_subset["mean_test_score"], label="Random Forest", marker="o")

# Plot Ensemble results
if ensemble_hyperparameter_columns:
    ensemble_subset = results[results["model"] == "ensemble"]
    if not ensemble_subset.empty:
        plt.plot(ensemble_subset["Ensemble_hyperparameter_set"], ensemble_subset["mean_test_score"], label="Ensemble", marker="s")

plt.title("Hyperparameter Tuning Results for Random Forest and Ensemble Models")
plt.xlabel("Hyperparameter Set")
plt.ylabel("Mean Test Score (Accuracy)")
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig(args.output)
print(f"Hyperparameter tuning plot saved to {args.output}")
