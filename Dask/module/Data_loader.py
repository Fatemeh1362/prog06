import dask.dataframe as dd
import pandas as pd
import time

class DataLoader:
    def __init__(self, gene_expression_path, clinical_data_path):
        """Initialize file paths."""
        self.gene_expression_path = gene_expression_path
        self.clinical_data_path = clinical_data_path

    def load_with_pandas(self, file_path, delimiter=None, file_type='csv', **kwargs):
        """Load data using Pandas."""
        start_time = time.time()
        if file_type == 'csv':
            df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
        elif file_type == 'excel':
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
        loading_time = time.time() - start_time
        return df, loading_time

    def load_with_dask(self, file_path, delimiter=None, file_type='csv', npartitions=4, **kwargs):
        """Load data using Dask."""
        start_time = time.time()
        if file_type == 'csv':
            df = dd.read_csv(file_path, delimiter=delimiter, **kwargs)
        elif file_type == 'excel':
            pandas_df = pd.read_excel(file_path, **kwargs)
            df = dd.from_pandas(pandas_df, npartitions=npartitions)
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
        loading_time = time.time() - start_time
        return df, loading_time

    def compare_memory_usage(self, dask_df, pandas_df):
        """Compare memory usage between Dask and Pandas DataFrames."""
        dask_memory = dask_df.memory_usage().compute().sum()
        pandas_memory = pandas_df.memory_usage().sum()
        return dask_memory / (1024 ** 2), pandas_memory / (1024 ** 2)  # Return in MB

    def print_comparison(self, pandas_time, dask_time, pandas_memory, dask_memory, data_type):
        """Print comparison results for a dataset."""
        print(f"\n--- {data_type} Data Comparison ---")
        print(f"Pandas Loading Time: {pandas_time:.2f} seconds")
        print(f"Dask Loading Time: {dask_time:.2f} seconds")
        print(f"Pandas Memory Usage: {pandas_memory:.2f} MB")
        print(f"Dask Memory Usage: {dask_memory:.2f} MB")



    