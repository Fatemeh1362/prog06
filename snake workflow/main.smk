# Configurations
output_dir = "output/"
data_dir = "data/"
models = ["random_forest", "ensemble"]

# The 'all' rule depends on all final outputs
rule all:
    input:
        "output/comparison_plot.png",
        "output/final_report.csv",
        "output/hyperparameter_plot.png",
        "output/RF_tuning_results.csv",
        "output/ensemble_tuning_results.csv",
        "output/RF_best_model.pkl",
        "output/ensemble_best_model.pkl"

rule preprocess_data:
    input:
        dataset="/homes/fmonfared/my_project/prog06/data/diseaseDk_PCR_Hanze_WP2.xlsx"
    output:
        train="data/train_data.csv",
        test="data/test_data.csv"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/preprocessor.py \
        --input {input.dataset} --output-train {output.train} --output-test {output.test}
        """

# Rule to tune the Random Forest model
rule tune_random_forest:
    input:
        train="data/train_data.csv",  # Depends on preprocess_data
        model="output/random_forest_model.pkl"
    output:
        results="output/RF_tuning_results.csv",
        model="output/RF_best_model.pkl"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/tune_RandomForest.py \
        --train {input.train} --model {input.model} --results {output.results} --best-model {output.model}
        """

# Rule to train the Ensemble model
rule train_ensemble:
    input:
        train="data/train_data.csv"  # Depends on preprocess_data
    output:
        model="output/ensemble_model.pkl"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/trainens.py \
        --train {input.train} --output {output.model}
        """

# Rule to tune the Ensemble model
rule tune_ensemble:
    input:
        train="data/train_data.csv",  # Depends on preprocess_data
        model="output/ensemble_model.pkl"
    output:
        results="output/ensemble_tuning_results.csv",
        model="output/ensemble_best_model.pkl"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/tune_ensemble.py \
        --input {input.train} --model {input.model} --results {output.results} --best-model {output.model}
        """

# Rule to evaluate the Random Forest model
rule evaluate_random_forest:
    input:
        model="output/RF_best_model.pkl",
        data="data/test_data.csv"  # Depends on preprocess_data
    output:
        results="output/random_forest_results.txt"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/evaluation.py \
        --test {input.data} --model {input.model} --output {output.results}
        """

# Rule to evaluate the Ensemble model
rule evaluate_ensemble:
    input:
        model="output/ensemble_best_model.pkl",
        data="data/test_data.csv"  # Depends on preprocess_data
    output:
        results="output/ensemble_results.txt"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/evaluation.py \
        --test {input.data} --model {input.model} --output {output.results}
        """

# Rule to generate the final report
rule create_report:
    input:
        rf="output/random_forest_results.txt",
        ensemble="output/ensemble_results.txt"
    output:
        report="output/final_report.csv"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/Create_report.py \
        --rf {input.rf} --ensemble {input.ensemble} --output {output.report}
        """

# Rule to generate a hyperparameter tuning plot
rule plot_hyperparameter_results:
    input:
        ensemble_tuning_results="output/ensemble_tuning_results.csv"
    output:
        plot="output/hyperparameter_plot.png"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/plot_hyperparameter_results.py \
        --input {input.ensemble_tuning_results} --output {output.plot}
        """

# Rule to generate a comparison plot
rule generate_comparison_plot:
    input:
        rf_metrics="output/random_forest_results.txt",
        ensemble_metrics="output/ensemble_results.txt"
    output:
        plot="output/comparison_plot.png"
    shell:
        """
        python /homes/fmonfared/my_project/prog06/scripts/generate_comparision_plot.py \
        --rf {input.rf_metrics} --ensemble {input.ensemble_metrics} --output {output.plot}
        """
