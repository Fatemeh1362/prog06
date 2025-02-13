import pandas as pd
import dask.dataframe as dd
import dask


class DelayedDataPreprocessor:

    def __init__(self, pandas_gene_df, pandas_clinical_df, npartitions=1, threshold=0.5):
        """
        Initializes the preprocessor with gene expression and clinical data.
        :param pandas_gene_df: Pandas DataFrame containing gene expression data.
        :param pandas_clinical_df: Pandas DataFrame containing clinical data.
        :param npartitions: Number of partitions for Dask conversion.
        :param threshold: Threshold for missing value filtering (fraction of expression columns).
        """
        if "!Sample_geo_accession" in pandas_gene_df.columns:
            pandas_gene_df = pandas_gene_df.rename(columns={"!Sample_geo_accession": "expression"})

        # Convert numeric columns in Pandas before Dask conversion
        self.expression_columns = [col for col in pandas_gene_df.columns if col != "expression"]
        pandas_gene_df[self.expression_columns] = pandas_gene_df[self.expression_columns].apply(pd.to_numeric, errors="coerce")

        # Convert Pandas → Dask (without `meta` argument)
        self.dask_gene_df = dd.from_pandas(pandas_gene_df.astype({"expression": "object"}), npartitions=npartitions)
        self.dask_clinical_df = dd.from_pandas(pandas_clinical_df, npartitions=npartitions)

        self.threshold = int(threshold * len(self.expression_columns))  # Convert fraction to absolute count

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

    def handle_missing_values_dask(self, df):
        """Handles missing values in gene expression data using Dask."""
        df = df.dropna(thresh=self.threshold)

        # Identify numeric columns
        numeric_cols = [col for col in df.columns if col != "expression"]

        # Fill missing values using mean per partition
        df = df.map_partitions(
            lambda df_part: df_part.assign(
                **{col: df_part[col].fillna(df_part[col].mean()) for col in numeric_cols}
            ),
            meta={col: "float64" for col in numeric_cols}
        )

        return df

    def handle_missing_values_clinical_dask(self, df):
        """Handles missing values in the clinical dataset using Dask."""
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        # Fill numeric columns with mean
        df = df.assign(**{
            col: df[col].map_partitions(lambda x: x.fillna(x.mean()), meta=("col", "float64"))
            for col in numeric_cols
        })

        # Compute mode for categorical columns once, then apply
        if len(categorical_cols) > 0:
            mode_df = df[categorical_cols].mode().compute()
            if not mode_df.empty:
                mode_vals = mode_df.iloc[0]  # Take the first mode value
                df[categorical_cols] = df[categorical_cols].fillna(mode_vals)


                return df

    def filter_low_quality_data_dask(self, df, threshold=0.1):
        """Filters low-quality genes in gene expression data using Dask."""
        expression_columns = [col for col in df.columns if col != "expression"]

        # Compute mean expression per gene within Dask to avoid Pandas conversion issues
        df["mean_expression"] = df[expression_columns].mean(axis=1, skipna=True)

        # Apply filtering in Dask using .map_partitions()
        df = df.map_partitions(lambda d: d[d["mean_expression"] >= threshold])

        # Drop the mean_expression column after filtering
        return df.drop(columns=["mean_expression"])


    def filter_low_quality_data_pandas(self, df, threshold=0.1):
        """Filters low-quality genes in gene expression data using Pandas."""
        df = df.copy()
        expression_columns = [col for col in df.columns if col != "expression"]

        # Convert expression columns to numeric to avoid string issues
        df[expression_columns] = df[expression_columns].apply(pd.to_numeric, errors="coerce")

        # Compute mean expression per gene
        mean_values = df[expression_columns].mean(axis=1, skipna=True)

        # Apply filtering
        df = df[mean_values >= threshold]

        return df
