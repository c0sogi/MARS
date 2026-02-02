import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library import config


class PreprocessingPipeline:
    """
    A wrapper class for the preprocessing pipeline to ensure high-precision
    transformations. It combines PowerTransformer (Yeo-Johnson) and StandardScaler,
    enforcing float64 dtype throughout.
    """

    def __init__(self):
        # Yeo-Johnson is used to stabilize variance and make data more Gaussian-like.
        # standardize=False because we apply StandardScaler explicitly afterwards.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the pipeline on the training data.
        X: numpy array of shape (n_samples, n_features)
        """
        X = X.astype(np.float64)
        # Fit PowerTransformer
        self.pt.fit(X)
        # Transform to fit StandardScaler
        X_pt = self.pt.transform(X)
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """
        Apply the learned transformations to new data.
        X: numpy array of shape (n_samples, n_features)
        """
        X = X.astype(np.float64)
        X_pt = self.pt.transform(X)
        X_scaled = self.scaler.transform(X_pt)
        return X_scaled


class LeafDataProcessor:
    """
    Handles loading, preprocessing, and caching of the leaf dataset.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.files = {
            "X_train": os.path.join(self.cache_dir, "X_train.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.npy"),
            "test_ids": os.path.join(self.cache_dir, "test_ids.npy"),
            "classes": os.path.join(self.cache_dir, "classes.npy"),
        }

    def load_data(self, load_cached_data=True):
        """
        Loads the processed dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from local cache.
                                     If False or cache miss, re-processes raw data.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
        """
        if load_cached_data and self._check_cache_exists():
            print(f"Loading cached data from {self.cache_dir}...")
            return self._load_from_cache()

        print("Processing data from scratch...")
        return self._process_and_cache()

    def _check_cache_exists(self):
        """Checks if all required cache files exist."""
        return all(os.path.exists(path) for path in self.files.values())

    def _load_from_cache(self):
        """Loads numpy arrays from the cache directory."""
        X_train = np.load(self.files["X_train"])
        y_train = np.load(self.files["y_train"])
        X_val = np.load(self.files["X_val"])
        y_val = np.load(self.files["y_val"])
        X_test = np.load(self.files["X_test"])
        test_ids = np.load(self.files["test_ids"])
        classes = np.load(self.files["classes"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    def _process_and_cache(self):
        """
        Reads raw CSVs, applies the preprocessing pipeline, and saves to cache.
        """
        # 1. Load Raw Data
        print("Loading CSV files...")
        df_train = pd.read_csv(config.TRAIN_CSV)
        df_val = pd.read_csv(config.VAL_CSV)
        df_test = pd.read_csv(config.TEST_CSV)

        # 2. Extract Features and Targets
        # Ensure features are strictly ordered as defined in config
        feature_cols = config.FEATURE_COLS

        # Convert to float64 immediately
        X_train_raw = df_train[feature_cols].values.astype(np.float64)
        X_val_raw = df_val[feature_cols].values.astype(np.float64)
        X_test_raw = df_test[feature_cols].values.astype(np.float64)

        y_train_raw = df_train[config.TARGET_COL].values
        y_val_raw = df_val[config.TARGET_COL].values
        test_ids = df_test[config.ID_COL].values

        # 3. Encode Targets
        print("Encoding targets...")
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_raw)
        y_val_enc = le.transform(y_val_raw)
        classes = le.classes_

        # 4. Apply Preprocessing Pipeline
        # Fit ONLY on training data, then transform all sets
        print("Fitting preprocessing pipeline (Yeo-Johnson + StandardScaler)...")
        pipeline = PreprocessingPipeline()
        pipeline.fit(X_train_raw)

        print("Transforming datasets...")
        X_train_processed = pipeline.transform(X_train_raw)
        X_val_processed = pipeline.transform(X_val_raw)
        X_test_processed = pipeline.transform(X_test_raw)

        # 5. Save to Cache
        print(f"Saving processed data to {self.cache_dir}...")
        np.save(self.files["X_train"], X_train_processed)
        np.save(self.files["y_train"], y_train_enc)
        np.save(self.files["X_val"], X_val_processed)
        np.save(self.files["y_val"], y_val_enc)
        np.save(self.files["X_test"], X_test_processed)
        np.save(self.files["test_ids"], test_ids)
        np.save(self.files["classes"], classes)

        return (
            X_train_processed,
            y_train_enc,
            X_val_processed,
            y_val_enc,
            X_test_processed,
            test_ids,
            classes,
        )
