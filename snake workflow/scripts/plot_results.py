import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Generate Accuracy Plot")
parser.add_argument("--input", required=True, help="Path to the final report CSV")
parser.add_argument("--output", required=True, help="Path to save the accuracy plot")
args = parser.parse_args()

# Load the final report CSV
print(f"Loading final report from {args.input}")
df = pd.read_csv(args.input)

# Inspect the columns
print("Columns in the CSV:", df.columns)

# Ensure necessary columns exist
if "Metric" not in df.columns or "Value" not in df.columns:
    print("Error: The final report must include 'Metric' and 'Value' columns.")
    exit(1)

# Filter accuracy-related metrics
accuracy_data = df[df["Metric"].str.contains("Accuracy", case=False, na=False)]

if accuracy_data.empty:
    print("Error: No accuracy data found in the final report.")
    exit(1)

# Plotting
plt.figure(figsize=(8, 6))
accuracy_data.plot(
    x="Unnamed: 0",  # Use the first column for models
    y="Value",  # Use the 'Value' column for accuracy values
    kind="bar",
    color="blue",
    edgecolor="black",
    legend=False,
    width=0.6,
    figsize=(10, 6),
)
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=45, ha="right")  # Rotate labels for better visibility
plt.ylim(0, 1)  # Accuracy values should be between 0 and 1
plt.tight_layout()

# Save the plot
print(f"Saving accuracy plot to {args.output}")
plt.savefig(args.output)
plt.show()
