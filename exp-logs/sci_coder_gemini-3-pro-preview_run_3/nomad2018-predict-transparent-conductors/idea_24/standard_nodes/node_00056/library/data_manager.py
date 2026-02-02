import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    RANDOM_SEED,
)
from library.feature_engineering import extract_features


class DataManager:
    """
    Handles data loading, preprocessing, feature matrix construction, and caching.
    """

    def __init__(self, working_dir=WORKING_DIR):
        self.working_dir = working_dir
        # Define cache paths for the final processed dataframes
        self.train_cache_path = os.path.join(working_dir, "processed_train.parquet")
        self.val_cache_path = os.path.join(working_dir, "processed_val.parquet")
        self.test_cache_path = os.path.join(working_dir, "processed_test.parquet")

    def _load_metadata(self):
        """Loads the metadata CSVs for train, val, and test splits."""
        if not os.path.exists(TRAIN_METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found: {TRAIN_METADATA_PATH}")

        df_train = pd.read_csv(TRAIN_METADATA_PATH)
        df_val = pd.read_csv(VAL_METADATA_PATH)
        df_test = pd.read_csv(TEST_METADATA_PATH)
        return df_train, df_val, df_test

    def _process_split(self, df_meta, split_name, load_cached_features=True):
        """
        Calls the feature engineering module to extract features for a given split.
        Merges targets if available.
        """
        print(f"Extracting features for {split_name} set...")
        # extract_features handles caching of the raw feature extraction step
        df_features = extract_features(df_meta, load_cached_data=load_cached_features)

        # Merge targets if they exist in metadata (train/val)
        if "formation_energy_ev_natom" in df_meta.columns:
            # Select only ID and targets to merge
            targets = df_meta[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]
            # Merge on ID
            df_merged = pd.merge(df_features, targets, on="id", how="left")
        else:
            df_merged = df_features

        return df_merged

    def load_and_process_data(self, load_cached_data=True):
        """
        Main function to load, process, clean, and cache all data splits.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.
                                     If False, re-runs the full processing pipeline.

        Returns:
            tuple: ((X_train, y_train), (X_val, y_val), (X_test, ids_test))
        """
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # 1. Try to load from cache
        if (
            load_cached_data
            and os.path.exists(self.train_cache_path)
            and os.path.exists(self.val_cache_path)
            and os.path.exists(self.test_cache_path)
        ):
            print(f"Loading processed data from cache in {self.working_dir}...")
            df_train = pd.read_parquet(self.train_cache_path)
            df_val = pd.read_parquet(self.val_cache_path)
            df_test = pd.read_parquet(self.test_cache_path)

        else:
            # 2. Compute from scratch
            print("Computing processed data from scratch...")

            # Load metadata
            meta_train, meta_val, meta_test = self._load_metadata()

            # Extract features (this step uses its own caching for the heavy geometry calc)
            df_train = self._process_split(
                meta_train, "train", load_cached_features=load_cached_data
            )
            df_val = self._process_split(
                meta_val, "val", load_cached_features=load_cached_data
            )
            df_test = self._process_split(
                meta_test, "test", load_cached_features=load_cached_data
            )

            # Target Transformation: log(1 + y)
            # Targets are non-negative (energies), so log1p is safe and stabilizes variance
            for df in [df_train, df_val]:
                df["formation_energy_log"] = np.log1p(df["formation_energy_ev_natom"])
                df["bandgap_energy_log"] = np.log1p(df["bandgap_energy_ev"])

            # Feature Cleaning
            # Identify feature columns (exclude IDs, targets, and file paths)
            exclude_cols = [
                "id",
                "formation_energy_ev_natom",
                "bandgap_energy_ev",
                "formation_energy_log",
                "bandgap_energy_log",
                "file_path",
            ]

            # Get all potential feature columns from training data
            feature_cols = [c for c in df_train.columns if c not in exclude_cols]

            # Drop constant columns (variance == 0)
            # We determine constants based on the Training set only to avoid leakage
            constant_cols = [c for c in feature_cols if df_train[c].nunique() <= 1]
            print(
                f"Dropping {len(constant_cols)} constant columns found in training set."
            )

            df_train = df_train.drop(columns=constant_cols)
            df_val = df_val.drop(columns=constant_cols, errors="ignore")
            df_test = df_test.drop(columns=constant_cols, errors="ignore")

            # Fill NaNs with 0
            # NaNs can appear if RDF bins are empty or topology features are undefined for some structures
            df_train = df_train.fillna(0.0)
            df_val = df_val.fillna(0.0)
            df_test = df_test.fillna(0.0)

            # Save final processed dataframes to cache
            print("Saving processed data to cache...")
            df_train.to_parquet(self.train_cache_path)
            df_val.to_parquet(self.val_cache_path)
            df_test.to_parquet(self.test_cache_path)

        # 3. Prepare Return Values (X and y separation)
        target_cols = ["formation_energy_log", "bandgap_energy_log"]
        exclude_cols = [
            "id",
            "formation_energy_ev_natom",
            "bandgap_energy_ev",
            "formation_energy_log",
            "bandgap_energy_log",
            "file_path",
        ]

        # Select features present in the dataframe
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

        # Ensure validation and test have the same columns as train (in same order)
        # This handles cases where test might have had a column dropped that wasn't in train, or order issues
        X_train = df_train[feature_cols]
        y_train = df_train[target_cols]

        X_val = df_val[feature_cols]
        y_val = df_val[target_cols]

        # Test set doesn't have targets
        X_test = df_test[feature_cols]
        ids_test = df_test["id"]

        print(f"Data processing complete. Feature shape: {X_train.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, ids_test)

    @staticmethod
    def inverse_transform(y_log):
        """
        Applies exp(y) - 1 to reverse the log1p transformation.
        Used for generating final submission predictions.
        """
        return np.expm1(y_log)
