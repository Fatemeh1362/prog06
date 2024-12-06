include: "/homes/fmonfared/my_project/prog06/rules/preprocess.smk"
include: "/homes/fmonfared/my_project/prog06/rules/train.smk"
include: "/homes/fmonfared/my_project/prog06/rules/tune.smk"
include: "/homes/fmonfared/my_project/prog06/rules/plot_hyper.smk"
include: "/homes/fmonfared/my_project/prog06/rules/evaluate.smk"
include: "/homes/fmonfared/my_project/prog06/rules/report.smk"
include: "/homes/fmonfared/my_project/prog06/rules/comparision.smk"

# Define the final rule with all outputs
rule workflow_complete:
    input:
        "results/RF_best_model.pkl",               # Tuned Random Forest model
        "results/ensemble_best_model.pkl",         # Tuned Ensemble model
        "results/RF_tuned_accuracy.txt",           # Accuracy of tuned RF model
        "results/ensemble_tuned_accuracy.txt",     # Accuracy of tuned ensemble model
        "results/ensemble_tuning_results.csv",     # Ensemble tuning results
        "results/RF_tuning_results.csv",           # RF tuning results
        "results/hyperparameter_plot.png",
        "results/RF_metrics.json",                 # Metrics of Random Forest model
        "results/ensemble_metrics.json",           # Metrics of Ensemble model
        "results/final_report.txt",                # Final report
        "results/comparison_plot.png"              # Accuracy comparison plot
