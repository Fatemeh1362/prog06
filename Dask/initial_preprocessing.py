import pandas as pd
import dask.dataframe as dd
import dask.delayed
import dask


class DelayedDataPreprocessor:

    def __init__(self, pandas_gene_df, pandas_clinical_df, npartitions=4, threshold=0.5):
        """
        Initializes the preprocessor with gene expression and clinical data.
        :param pandas_gene_df: Pandas DataFrame containing gene expression data.
        :param pandas_clinical_df: Pandas DataFrame containing clinical data.
        :param npartitions: Number of partitions for Dask conversion.
        :param threshold: Threshold for missing value filtering (fraction of expression columns).
        """
        # ✅ Rename '!Sample_geo_accession' to 'expression'
        if "!Sample_geo_accession" in pandas_gene_df.columns:
            pandas_gene_df = pandas_gene_df.rename(columns={"!Sample_geo_accession": "expression"})

        self.pandas_gene_df = pandas_gene_df.copy()
        self.pandas_clinical_df = pandas_clinical_df.copy()
        self.npartitions = npartitions
        self.threshold = int(threshold * (len(pandas_gene_df.columns) - 1))  # Convert fraction to absolute count
        self.expression_columns = [col for col in pandas_gene_df.columns if col != "expression"]  # Only numeric columns

    @dask.delayed
    def preprocess_pandas(self):
        """Cleans and preprocesses the gene expression dataset using Pandas (Delayed Execution)."""
        df = self.pandas_gene_df.copy()
        
        # ✅ Ensure 'expression' column exists
        if "expression" in df.columns:
            df = df[~df["expression"].str.startswith("!")].reset_index(drop=True)

        # ✅ Convert expression columns to numeric
        df[self.expression_columns] = df[self.expression_columns].apply(pd.to_numeric, errors="coerce")

        # ✅ Drop rows where more than `threshold`% of expression values are NaN
        df = df.dropna(thresh=self.threshold)

        # ✅ Fill missing values with column means (only for numeric expression columns)
        df[self.expression_columns] = df[self.expression_columns].apply(lambda x: x.fillna(x.mean()), axis=0)

        return df

    @dask.delayed
    def convert_to_dask(self, df):
        """Converts a cleaned Pandas DataFrame to a Dask DataFrame (Delayed Execution)."""
        return dd.from_pandas(df, npartitions=self.npartitions)

    @dask.delayed
    def handle_missing_values_dask(self, df):
        """Handles missing values in gene expression data using Dask (Delayed Execution)."""
        df = df.dropna(thresh=self.threshold)

        # ✅ Ensure numeric columns before applying mean
        numeric_cols = [col for col in df.columns if col != "expression"]

        # ✅ Define proper metadata
        meta_dict = {"expression": "object"}
        meta_dict.update({col: "float64" for col in numeric_cols})

        # ✅ Fill missing values only for numeric columns
        df = df.map_partitions(
            lambda df_part: df_part.assign(
                **{col: df_part[col].astype(float).fillna(df_part[col].astype(float).mean()) for col in numeric_cols}
            ),
            meta=meta_dict
        )

        return df

    def handle_missing_values_pandas(self, df, method="drop"):
        """Handles missing values in Pandas (Clinical or Gene Data)."""
        df = df.copy()
        
        if method == "drop":
            return df.dropna()
        elif method == "mean":
            return df.fillna(df.select_dtypes(include=["number"]).mean())  # Apply mean only to numeric columns
        elif method == "mode":
            return df.fillna(df.mode().iloc[0])
        else:
            raise ValueError("Method must be 'drop', 'mean', or 'mode'.")

    @dask.delayed
    def handle_missing_values_clinical_dask(self, df):
        """Handles missing values in the clinical dataset using Dask (Delayed Execution)."""
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        # ✅ Fill numeric columns with mean values using assign()
        df = df.assign(**{
            col: df[col].map_partitions(lambda x: x.fillna(x.mean()), meta=("col", "float64"))
            for col in numeric_cols
        })

        # ✅ Fill categorical columns with mode values
        for col in categorical_cols:
            mode_val = df[col].mode().compute()[0] if not df[col].mode().compute().empty else None
            if mode_val:
                df[col] = df[col].fillna(mode_val)

        return df

        

    @dask.delayed
    def filter_low_quality_data_dask(self, df, threshold=0.1):
        """Filters low-quality genes in gene expression data using Dask (Delayed Execution)."""
        expression_columns = [col for col in df.columns if col != "expression"]

        # ✅ Ensure numeric types for expression columns before mean calculation
        df[expression_columns] = df[expression_columns].astype("float64")

        # ✅ Compute mean expression per gene
        df["mean_expression"] = df[expression_columns].mean(axis=1, skipna=True)

        return df[df["mean_expression"] >= threshold].drop(columns=["mean_expression"])

    def filter_low_quality_data_pandas(self, df, threshold=0.1):
        
        df = df.copy()
        expression_columns = [col for col in df.columns if col != "expression"]

        # ✅ Ensure numeric types for expression columns
        df[expression_columns] = df[expression_columns].apply(pd.to_numeric, errors="coerce")

        # ✅ Compute mean expression per gene
        df["mean_expression"] = df[expression_columns].mean(axis=1, skipna=True)

        return df[df["mean_expression"] >= threshold].drop(columns=["mean_expression"])
