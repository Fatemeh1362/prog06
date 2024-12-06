import argparse
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser(description="Generate final report")
parser.add_argument("--rf", required=True, help="Path to Random Forest results file")
parser.add_argument("--ensemble", required=True, help="Path to Ensemble results file")
parser.add_argument("--output", required=True, help="Path to save the final report")
args = parser.parse_args()

# Load results
print(f"Loading Random Forest results from {args.rf}")
rf_results = pd.read_csv(args.rf, header=None, names=["Metric", "Value"])

print(f"Loading Ensemble results from {args.ensemble}")
ensemble_results = pd.read_csv(args.ensemble, header=None, names=["Metric", "Value"])

# Combine results
print("Combining results...")
final_report = pd.concat([rf_results, ensemble_results], keys=["Random Forest", "Ensemble"])

# Save final report
print(f"Saving final report to {args.output}")
final_report.to_csv(args.output)

print("Final report generation completed successfully.")
