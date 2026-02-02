import os
import numpy as np
import pandas as pd
from library.config import Config
from library.descriptors import generate_features


class DataPipeline:
    """
    Orchestrates the data loading, feature extraction, and cleaning pipeline.
    Utilizes the FeatureExtractor from library.descriptors to process crystal structures.
    """

    def __init__(self):
        self.config = Config

    def load_data(self, load_cached_data=True):
        """
        Loads train, validation, and test data.
        Generates features if not cached, or loads from cache.
        Performs data cleaning (dropping constant columns).

        Args:
            load_cached_data (bool): If True, attempts to load from Parquet cache.
                                     If False, regenerates features.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        # Load Metadata
        if not os.path.exists(self.config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Metadata not found at {self.config.TRAIN_METADATA_PATH}"
            )

        train_meta = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        val_meta = pd.read_csv(self.config.VAL_METADATA_PATH)
        test_meta = pd.read_csv(self.config.TEST_METADATA_PATH)

        # Generate or Load Features
        # The generate_features function in library.descriptors handles the caching logic
        # strictly as required: checks file, loads if exists and flag is True, else computes and saves.

        print("Processing Training Set...")
        train_df = generate_features(
            train_meta,
            self.config.TRAIN_FEATURES_PATH,
            load_cached_data=load_cached_data,
        )

        print("Processing Validation Set...")
        val_df = generate_features(
            val_meta, self.config.VAL_FEATURES_PATH, load_cached_data=load_cached_data
        )

        print("Processing Test Set...")
        test_df = generate_features(
            test_meta, self.config.TEST_FEATURES_PATH, load_cached_data=load_cached_data
        )

        # Clean Data (Drop constant columns based on training set)
        train_df, val_df, test_df = self.clean_data(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def clean_data(self, train_df, val_df, test_df):
        """
        Drops columns that are constant in the training set from all datasets.
        Fills missing values with 0.
        """
        print("Cleaning data...")

        # Fill NaNs with 0 (assuming 0 is appropriate for missing structural features like RDF bins)
        train_df = train_df.fillna(0)
        val_df = val_df.fillna(0)
        test_df = test_df.fillna(0)

        # Identify constant columns in training data
        # Exclude metadata/target columns from being dropped even if constant (unlikely for targets)
        # 'id' is necessary for submission mapping
        # 'file_path' is string, likely not constant but we exclude it from check

        protected_cols = ["id", "file_path"] + self.config.TARGET_COLS

        # Select numeric columns for check
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns

        constant_cols = []
        for col in numeric_cols:
            if col in protected_cols:
                continue

            # Check if std is 0 (or all values are the same)
            if train_df[col].std() == 0:
                constant_cols.append(col)

        if constant_cols:
            print(f"Dropping {len(constant_cols)} constant columns.")
            train_df = train_df.drop(columns=constant_cols)
            val_df = val_df.drop(columns=constant_cols, errors="ignore")
            test_df = test_df.drop(columns=constant_cols, errors="ignore")
        else:
            print("No constant columns found.")

        return train_df, val_df, test_df
