import os
import random
import numpy as np
import pandas as pd
from library.feature_extraction import generate_features

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class CrystalDataHandler:
    """
    Handles data loading, feature generation, and preprocessing for crystal structures.
    Uses caching to avoid re-computing features on every run.
    """

    def __init__(self, metadata_dir="./metadata", cache_dir="./working/idea_10"):
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        self.train_meta_path = os.path.join(metadata_dir, "train_metadata.csv")
        self.val_meta_path = os.path.join(metadata_dir, "val_metadata.csv")
        self.test_meta_path = os.path.join(metadata_dir, "test_metadata.csv")

    def load_data(self, load_cached_data=True, max_samples=None):
        """
        Loads training, validation, and test data.
        Generates features if not cached using the provided library functions.
        Applies log(1+x) transformation to target variables.

        Args:
            load_cached_data (bool): Whether to load from parquet cache if available.
            max_samples (int, optional): If set, truncates the datasets to this many samples
                                         (useful for debugging/testing).

        Returns:
            tuple: ((X_train, y_train), (X_val, y_val), X_test, test_ids)
        """
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Generate or load features using the library function
        # This function handles the iteration through IDs and caching logic internally
        df_train, df_val, df_test = generate_features(
            train_meta_path=self.train_meta_path,
            val_meta_path=self.val_meta_path,
            test_meta_path=self.test_meta_path,
            output_dir=self.cache_dir,
            load_cached_data=load_cached_data,
        )

        # Apply subsampling if requested (for debugging)
        if max_samples is not None:
            print(f"Subsampling datasets to {max_samples} samples.")
            df_train = df_train.head(max_samples)
            df_val = df_val.head(max_samples)
            df_test = df_test.head(max_samples)

        # Define target columns
        target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

        # Define metadata columns to exclude from feature set
        # 'id' and 'file_path' are identifiers/metadata, not input features
        meta_cols = ["id", "file_path"]

        # Prepare Training Data
        # Drop metadata and targets from X
        X_train = df_train.drop(
            columns=[c for c in meta_cols + target_cols if c in df_train.columns],
            errors="ignore",
        )
        # Apply log(1+x) transformation to targets
        y_train = df_train[target_cols].apply(np.log1p)

        # Prepare Validation Data
        X_val = df_val.drop(
            columns=[c for c in meta_cols + target_cols if c in df_val.columns],
            errors="ignore",
        )
        y_val = df_val[target_cols].apply(np.log1p)

        # Prepare Test Data
        # Test data should not have targets, but we ensure they are dropped if present
        X_test = df_test.drop(
            columns=[c for c in meta_cols + target_cols if c in df_test.columns],
            errors="ignore",
        )
        test_ids = df_test["id"]

        print(f"Data loaded successfully.")
        print(
            f"Train features shape: {X_train.shape}, Train targets shape: {y_train.shape}"
        )
        print(
            f"Val features shape:   {X_val.shape}, Val targets shape:   {y_val.shape}"
        )
        print(f"Test features shape:  {X_test.shape}")

        return (X_train, y_train), (X_val, y_val), X_test, test_ids
