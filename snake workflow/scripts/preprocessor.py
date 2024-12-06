import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
import os
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_and_split(input_path, train_output, test_output):
    try:
        logging.info("Starting preprocessing...")

        # Check if input file exists
        if not os.path.exists(input_path):
            logging.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load the dataset
        logging.info(f"Loading dataset from {input_path}")
        data = pd.read_excel(input_path)

        # Impute missing 'strains_5' using 'strains_3' and 'strains_4'
        logging.info("Imputing missing 'strains_5' values...")
        data = data.dropna(subset=['strains_3', 'strains_4'])
        data['strains_3'] = data['strains_3'].fillna(data['strains_3'].mode()[0]).astype(int)
        data['strains_4'] = data['strains_4'].fillna(data['strains_4'].mode()[0]).astype(int)

        train_data_5 = data[data['strains_5'].notna()]
        test_data_5 = data[data['strains_5'].isna()]

        features_5 = ['strains_3', 'strains_4']
        X_train_5 = train_data_5[features_5]
        y_train_5 = train_data_5['strains_5']

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_5, y_train_5, test_size=0.2, random_state=42
        )

        regressor = RandomForestRegressor(random_state=42)
        regressor.fit(X_train_split, y_train_split)

        if not test_data_5.empty:
            X_test_5 = test_data_5[features_5]
            predictions_5 = regressor.predict(X_test_5)
            data.loc[data['strains_5'].isna(), 'strains_5'] = predictions_5.round().astype(int)

        data['strains_5'] = data['strains_5'].astype(int)

        # Impute other strain columns
        logging.info("Imputing other strain columns...")
        strain_columns = [f'strains_{i}' for i in range(1, 10) if i != 5]
        for col in strain_columns:
            data[col] = data[col].fillna(0).astype(int)

        # Split the data into train and test sets
        logging.info("Splitting data into train and test sets...")
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        # Save the train and test data
        os.makedirs(os.path.dirname(train_output), exist_ok=True)
        os.makedirs(os.path.dirname(test_output), exist_ok=True)

        logging.info(f"Saving train data to {train_output} with {len(train_data)} rows")
        train_data.to_csv(train_output, index=False)

        logging.info(f"Saving test data to {test_output} with {len(test_data)} rows")
        test_data.to_csv(test_output, index=False)

        logging.info("Preprocessing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split dataset")
    parser.add_argument("--input", required=True, help="Path to the input Excel file")
    parser.add_argument("--output-train", required=True, help="Path to save the train CSV")
    parser.add_argument("--output-test", required=True, help="Path to save the test CSV")

    args = parser.parse_args()

    preprocess_and_split(args.input, args.output_train, args.output_test)
