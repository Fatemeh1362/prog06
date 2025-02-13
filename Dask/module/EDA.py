import time
import pandas as pd
import dask.dataframe as dd
import dask.delayed
import dask
import matplotlib.pyplot as plt
import seaborn as sns


class EDAProcessorDask:
    def __init__(self, clinical_data, gene_data):
        """
        Initialize the EDA processor with clinical and gene expression datasets.
        Converts Pandas DataFrames to Dask DataFrames if necessary.
        """
        self.clinical_data = (
            dd.from_pandas(clinical_data, npartitions=4).persist()
            if isinstance(clinical_data, pd.DataFrame) else clinical_data.persist()
        )
        self.gene_data = (
            dd.from_pandas(gene_data, npartitions=4).persist()
            if isinstance(gene_data, pd.DataFrame) else gene_data.persist()
        )


    @dask.delayed
    def compute_summary_statistics(self):
        """
        Compute summary statistics for clinical and gene expression datasets.
        """
        print("\n Computing Summary Statistics...")
        return self.clinical_data.describe(), self.gene_data.describe()

    
    
    @dask.delayed
    def identify_top_variable_genes(self, top_n=10):
        """
        Identify the top N most variable genes (genes with the highest standard deviation across patients).
        """
        print(f"\n Identifying Top {top_n} Most Variable Genes...")

        #  Ensure numeric values & remove NaNs before computing std
        expression_columns = [col for col in self.gene_data.columns if col != "expression"]
        valid_gene_data = self.gene_data[expression_columns].fillna(0)  # Fill NaNs

        gene_variability = valid_gene_data.std(axis=0).compute()

        #  Remove NaN values from standard deviation calculations
        gene_variability = gene_variability.dropna()

        #  Get top N variable genes
        top_genes = gene_variability.nlargest(top_n)
        
        return top_genes




    @dask.delayed
    def group_patients_by_variable(self, group_by="characteristics.tag.histology", selected_genes=None):
        """
        Group patients by a clinical variable and compute the mean expression of selected genes.
        """
        print(f"\n Grouping Patients by {group_by} and Computing Mean Expression...")

        # Ensure the grouping variable exists
        if group_by not in self.clinical_data.columns:
            raise KeyError(f" Error: Column '{group_by}' not found in clinical dataset!")

        # Select numeric gene expression columns
        if selected_genes is None:
            selected_genes = self.gene_data.columns[:5]  # Default: Use first 5 genes

        #  Merge Clinical Data into Gene Data
        merged_df = self.gene_data.merge(self.clinical_data[[group_by]], left_index=True, right_index=True)

        #  Compute Mean Expression by Group
        grouped_expression = merged_df.groupby(group_by)[selected_genes].mean()
        return grouped_expression

    def visualize_target_distribution(self, target_column="characteristics.tag.histology"):
        """
        Visualize the distribution of the target variable (TumorSubtype).
        """
        print(f"\n Visualizing Distribution of {target_column}...")

        # Ensure the column exists
        if target_column not in self.clinical_data.columns:
            raise KeyError(f" Error: Column '{target_column}' not found in clinical dataset!")

        # Compute value counts
        target_distribution = self.clinical_data[target_column].value_counts().compute()

        # Convert to Pandas for visualization
        target_distribution = target_distribution.to_frame().reset_index()
        target_distribution.columns = [target_column, "Count"]

        # Plot the distribution
        plt.figure(figsize=(10, 5))
        sns.barplot(x=target_column, y="Count", data=target_distribution, palette="viridis")
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {target_column}")
        plt.show()

    def run_pipeline(self):
        """
        Run the full EDA pipeline with Dask Delayed and compare performance with Pandas.
        """
        print("\n Running EDA Pipeline...")

        #  Compute Summary Statistics
        start_dask = time.time()
        summary_results = dask.compute(self.compute_summary_statistics())[0]  #  Compute first
        clinical_summary_dask, gene_summary_dask = summary_results  #  Unpack after computing

        dask_summary_time = time.time() - start_dask

        #  Compute Summary Statistics (Pandas)
        start_pandas = time.time()
        clinical_summary_pandas = self.clinical_data.compute().describe()
        gene_summary_pandas = self.gene_data.compute().describe()
        pandas_summary_time = time.time() - start_pandas

        print("\n Summary Statistics Computed!")
        print(f" Pandas Computation Time: {pandas_summary_time:.4f}s")
        print(f" Dask Computation Time: {dask_summary_time:.4f}s")

        #  Compute Top 10 Most Variable Genes
        start_dask = time.time()
        top_genes_dask = dask.compute(self.identify_top_variable_genes())[0]
        dask_top_genes_time = time.time() - start_dask

        #  Print the top 10 most variable genes
        print("\n Top 10 Most Variable Genes (Dask):")
        print(top_genes_dask)

        dask_top_genes_time = time.time() - start_dask

        start_pandas = time.time()
        top_genes_pandas = self.gene_data.compute().std(axis=0).nlargest(10)
        pandas_top_genes_time = time.time() - start_pandas

        print("\n Top 10 Most Variable Genes Computed!")
        print(f" Pandas Computation Time: {pandas_top_genes_time:.4f}s")
        print(f" Dask Computation Time: {dask_top_genes_time:.4f}s")

        #  Group by TumorSubtype & Compute Mean Expression
        start_dask = time.time()
        grouped_dask = dask.compute(self.group_patients_by_variable())[0]
        dask_grouping_time = time.time() - start_dask

        start_pandas = time.time()
        grouped_pandas = self.gene_data.compute().groupby(self.clinical_data.compute()["characteristics.tag.histology"]).mean()
        pandas_grouping_time = time.time() - start_pandas

        print("\n Grouped Expression Computed!")
        print(f" Pandas Computation Time: {pandas_grouping_time:.4f}s")
        print(f" Dask Computation Time: {dask_grouping_time:.4f}s")

        #  Visualize Tumor Subtype Distribution
        self.visualize_target_distribution()

        return {
            "clinical_summary_pandas": clinical_summary_pandas,
            "gene_summary_pandas": gene_summary_pandas,
            "clinical_summary_dask": clinical_summary_dask,
            "gene_summary_dask": gene_summary_dask,
            "top_genes_pandas": top_genes_pandas,
            "top_genes_dask": top_genes_dask,
            "grouped_pandas": grouped_pandas,
            "grouped_dask": grouped_dask
        }
