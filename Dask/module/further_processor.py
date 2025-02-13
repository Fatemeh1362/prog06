import pandas as pd
import dask.dataframe as dd
import dask.array as da
import dask.delayed
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class FurtherProcessor:
    def __init__(self, clinical_data, gene_data):
        """
        Initialize processor with clinical and gene expression datasets.
        Converts Pandas DataFrames to Dask DataFrames if necessary.
        """
        #  Convert to Dask if Pandas
        self.clinical_data = (
            dd.from_pandas(clinical_data, npartitions=4)
            if isinstance(clinical_data, pd.DataFrame) else clinical_data
        )
        self.gene_data = (
            dd.from_pandas(gene_data, npartitions=4)
            if isinstance(gene_data, pd.DataFrame) else gene_data
        )

        #  Rename `!Sample_geo_accession` to `expression` before processing
        if "!Sample_geo_accession" in self.gene_data.columns:
            print(" Renaming `!Sample_geo_accession` to `expression`...")
            self.gene_data = self.gene_data.rename(columns={"!Sample_geo_accession": "expression"})
                
    def normalize_gene_expression(self):
        """
        Normalize gene expression values using Z-score scaling.
        """
        print("\n Normalizing Gene Expression Data...")

        #  Drop "expression" column explicitly
        if "expression" in self.gene_data.columns:
            print(" Dropping 'expression' column before normalization...")
            self.gene_data = self.gene_data.drop(columns=["expression"])

        #  Select only numeric columns
        expression_columns = [col for col in self.gene_data.columns]

        #  Define correct metadata
        meta_dict = {col: "float64" for col in expression_columns}

        #  Ensure missing values are handled
        self.gene_data = self.gene_data.map_partitions(lambda df: df.fillna(0), meta=meta_dict)

        #  Compute Mean & Standard Deviation for Each Gene
        start_time = time.time()
        means = self.gene_data[expression_columns].mean().compute()
        stds = self.gene_data[expression_columns].std().compute()

        #  Avoid division by zero (Replace `std=0` with 1)
        stds[stds == 0] = 1

        #  Ensure "expression" is not in the DataFrame
        if "expression" in means.index:
            print(" ERROR: 'expression' is still present after dropping! Removing manually...")
            means = means.drop("expression")
            stds = stds.drop("expression")

        #  Apply Z-score Normalization Using Dask
        self.gene_data = self.gene_data.map_partitions(
            lambda df: df.assign(**{col: (df[col] - means[col]) / stds[col] for col in expression_columns}),
            meta=meta_dict  # Explicitly define metadata
        )

        end_time = time.time()
        print(f" Gene Expression Normalized - Time Taken: {end_time - start_time:.4f}s")

        return self.gene_data


    def encode_target_variable(self):
        """
        Encode the TumorSubtype variable into 0 (Non-Squamous) and 1 (Squamous).
        """
        print("\n Encoding TumorSubtype as Binary...")

        #  Ensure column exists
        if "characteristics.tag.histology" not in self.clinical_data.columns:
            raise KeyError(" Error: `characteristics.tag.histology` column not found in clinical dataset.")

        #  Create Encoding Function
        def encode_tumor_subtype(df):
            df = df.copy()  # Prevents modifying partitions directly
            df["TumorSubtype"] = df["characteristics.tag.histology"].astype(str).str.lower()
            df["TumorSubtype_encoded"] = df["TumorSubtype"].apply(
                lambda x: 1 if "squamous" in x else 0
            ).astype("int64")  # Ensure integer dtype
            return df

        #  Generate Correct Metadata Including All Original Columns
        meta_dict = {col: self.clinical_data[col].dtype for col in self.clinical_data.columns}
        meta_dict["TumorSubtype"] = "object"  # Ensure it's included
        meta_dict["TumorSubtype_encoded"] = "int64"  # Add encoded target variable

        #  Apply Encoding Using Correct Metadata
        self.clinical_data = self.clinical_data.map_partitions(encode_tumor_subtype, meta=meta_dict)

        return self.clinical_data





    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and test sets using Dask.
        """
        print("\n Splitting Data into Training & Test Sets...")

        #  Merge Clinical Data with Gene Expression Data
        merged_data = self.gene_data.merge(self.clinical_data, left_index=True, right_index=True)

        #  Convert to Pandas before splitting
        merged_data = merged_data.compute()
        #  Display first few rows
        print("\n Merged Data Sample:")
        print(merged_data.head())  # Display first few rows

        #  Save to CSV (Optional for debugging)
        merged_data.to_csv("merged_data_debug.csv", index=False)
        print(" Merged data saved to `merged_data_debug.csv`")
            #  Define Features (X) and Target (y)
        if "TumorSubtype_encoded" not in merged_data.columns:
            raise KeyError(" Error: 'TumorSubtype_encoded' column not found. Ensure encoding step is run first.")

        X = merged_data.drop(columns=["TumorSubtype", "TumorSubtype_encoded"], errors="ignore")
        y = merged_data["TumorSubtype_encoded"]

        #  Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print(f" Data Split Completed: {len(X_train)} training samples, {len(X_test)} test samples")

        return X_train, X_test, y_train, y_test


    def visualize_target_distribution(self, target_column="TumorSubtype_encoded"):
        """
        Visualize the distribution of the target variable.
        """
        print(f"\n Visualizing Distribution of {target_column}...")

        #  Compute value counts
        target_distribution = self.clinical_data[target_column].value_counts().compute()

        #  Convert to Pandas for visualization
        target_distribution = target_distribution.to_frame().reset_index()
        target_distribution.columns = [target_column, "Count"]

        #  Plot the distribution
        plt.figure(figsize=(8, 5))
        plt.bar(target_distribution[target_column], target_distribution["Count"], color="skyblue")
        plt.title(f"Distribution of {target_column}")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.xlabel(target_column)
        plt.show()
