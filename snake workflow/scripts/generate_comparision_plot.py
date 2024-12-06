import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Generate Comparison Plot")
parser.add_argument("--rf", required=True, help="Path to Random Forest results file")
parser.add_argument("--ensemble", required=True, help="Path to Ensemble results file")
parser.add_argument("--output", required=True, help="Path to save comparison plot")
args = parser.parse_args()

# Load model results
print(f"Loading Random Forest metrics from {args.rf}")
with open(args.rf, "r") as f:
    rf_metrics = dict(line.strip().split(": ") for line in f)

print(f"Loading Ensemble metrics from {args.ensemble}")
with open(args.ensemble, "r") as f:
    ensemble_metrics = dict(line.strip().split(": ") for line in f)

# Prepare data for plotting
data = []
for metric in rf_metrics.keys():
    data.append({"Model": "Random Forest", "Metric": metric, "Value": float(rf_metrics[metric])})
    data.append({"Model": "Ensemble", "Metric": metric, "Value": float(ensemble_metrics[metric])})

df = pd.DataFrame(data)

# Pivot for grouped bar plot
pivot_data = df.pivot(index="Metric", columns="Model", values="Value")
pivot_data.plot(kind="bar", figsize=(10, 6))

# Customize plot
plt.title("Model Comparison: Random Forest vs Ensemble")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model")
plt.tight_layout()

# Save plot
print(f"Saving comparison plot to {args.output}")
plt.savefig(args.output)
plt.show()
