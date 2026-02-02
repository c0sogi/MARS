import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.config import (
    METADATA_DIR,
    CACHE_DIR,
    NUMERIC_DTYPE,
    TABULAR_PREFIXES,
    GEOMETRIC_FEATURES,
    TARGET_COL,
    ID_COL,
    USE_YEO_JOHNSON,
    USE_STANDARD_SCALER,
    YEO_JOHNSON_STANDARDIZE,
    SEED,
)
from library.utils import compute_config_hash
from library.image_processing import process_dataset_images


class DataManager:
    def __init__(self, load_cached_data=True):
        """
        Initializes the DataManager.

        Args:
            load_cached_data (bool): Whether to attempt loading data from the cache.
        """
        self.load_cached_data = load_cached_data
        self.cache_dir = CACHE_DIR

        # Create a configuration dictionary to track changes in pipeline parameters
        self.config = {
            "geometric_features": GEOMETRIC_FEATURES,
            "tabular_prefixes": TABULAR_PREFIXES,
            "use_yeo_johnson": USE_YEO_JOHNSON,
            "use_standard_scaler": USE_STANDARD_SCALER,
            "yeo_johnson_standardize": YEO_JOHNSON_STANDARDIZE,
            "dtype": str(np.dtype(NUMERIC_DTYPE)),
            "seed": SEED,
        }
        self.config_hash = compute_config_hash(self.config)

    def load_data(self):
        """
        Loads the processed training, validation, and test datasets.
        Uses caching to avoid re-computation if the configuration hasn't changed.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
        """
        # Define expected cache file paths
        cache_files = {
            "X_train": os.path.join(self.cache_dir, "X_train.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.npy"),
            "test_ids": os.path.join(self.cache_dir, "test_ids.npy"),
            "classes": os.path.join(self.cache_dir, "classes.npy"),
            "hash": os.path.join(self.cache_dir, "config_hash.txt"),
        }

        # 1. Check Cache Validity
        cache_valid = False
        if self.load_cached_data and os.path.exists(cache_files["hash"]):
            try:
                with open(cache_files["hash"], "r") as f:
                    saved_hash = f.read().strip()
                if saved_hash == self.config_hash:
                    # Verify all data files exist
                    if all(
                        os.path.exists(p) for k, p in cache_files.items() if k != "hash"
                    ):
                        cache_valid = True
            except Exception:
                cache_valid = False

        # 2. Load from Cache
        if cache_valid:
            print(f"Loading preprocessed data from cache ({self.cache_dir})...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"])
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"])
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes

        # 3. Compute from Scratch
        print("Cache miss or invalid configuration. Processing data from scratch...")

        # Load and merge raw data
        df_train = self._load_and_merge("train")
        df_val = self._load_and_merge("val")
        df_test = self._load_and_merge("test")

        # Construct feature column list
        # 192 Tabular features: margin1..64, shape1..64, texture1..64
        tabular_cols = []
        for prefix in TABULAR_PREFIXES:
            for i in range(1, 65):
                tabular_cols.append(f"{prefix}{i}")

        # Combine with geometric features and sort alphanumerically
        feature_cols = sorted(tabular_cols + GEOMETRIC_FEATURES)

        print(f"Total Features: {len(feature_cols)}")

        # Extract Features (X)
        X_train = df_train[feature_cols].values.astype(NUMERIC_DTYPE)
        X_val = df_val[feature_cols].values.astype(NUMERIC_DTYPE)
        X_test = df_test[feature_cols].values.astype(NUMERIC_DTYPE)

        # Extract Targets (y) and IDs
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[TARGET_COL])
        y_val = le.transform(df_val[TARGET_COL])
        classes = le.classes_

        test_ids = df_test[ID_COL].values

        # 4. Inductive Preprocessing Pipeline
        # Fit transformers ONLY on training data, apply to all

        # Yeo-Johnson Power Transformation
        if USE_YEO_JOHNSON:
            print("Applying Yeo-Johnson Power Transformation...")
            # standardize=False allows StandardScaler to handle scaling separately
            pt = PowerTransformer(
                method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE
            )
            X_train = pt.fit_transform(X_train)
            X_val = pt.transform(X_val)
            X_test = pt.transform(X_test)

        # Standard Scaling
        if USE_STANDARD_SCALER:
            print("Applying Standard Scaling...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        # Enforce Double Precision (float64)
        X_train = X_train.astype(NUMERIC_DTYPE)
        X_val = X_val.astype(NUMERIC_DTYPE)
        X_test = X_test.astype(NUMERIC_DTYPE)

        # 5. Save to Cache
        print(f"Saving processed data to cache ({self.cache_dir})...")
        os.makedirs(self.cache_dir, exist_ok=True)

        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["X_val"], X_val)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["test_ids"], test_ids)
        np.save(cache_files["classes"], classes)

        with open(cache_files["hash"], "w") as f:
            f.write(self.config_hash)

        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    def _load_and_merge(self, mode):
        """
        Loads metadata, triggers geometric feature extraction, and merges datasets.

        Args:
            mode (str): 'train', 'val', or 'test'

        Returns:
            pd.DataFrame: Merged dataframe containing all features.
        """
        metadata_path = os.path.join(METADATA_DIR, f"{mode}.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata (contains ID, Target, Tabular Features)
        df_meta = pd.read_csv(metadata_path)

        # Load/Extract Geometric Features
        # This function handles its own caching of the raw extraction process
        df_geo = process_dataset_images(
            metadata_path, load_cached_data=self.load_cached_data
        )

        # Merge on ID
        # Inner join ensures we only keep rows where we have both metadata and image features
        df_merged = pd.merge(df_meta, df_geo, on=ID_COL, how="inner")

        # Verify merge integrity
        if len(df_merged) != len(df_meta):
            raise ValueError(
                f"Merge mismatch for {mode} set. "
                f"Metadata rows: {len(df_meta)}, Merged rows: {len(df_merged)}. "
                "Some images might be missing or failed processing."
            )

        return df_merged
