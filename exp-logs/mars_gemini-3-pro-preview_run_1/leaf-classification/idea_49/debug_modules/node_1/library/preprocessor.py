import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import Config
import library.feature_extraction as fe


class HighPrecisionPipeline:
    """
    A preprocessing pipeline that enforces float64 precision and applies
    Yeo-Johnson transformation followed by Standard Scaling.

    This pipeline ensures that all operations are performed in high precision
    to avoid the 10^-7 metric floor issues associated with float32.
    """

    def __init__(self):
        self.pt = None
        self.scaler = None

        # Initialize transformers based on Config
        if Config.APPLY_POWER_TRANSFORM:
            # Yeo-Johnson supports positive and negative values.
            # standardize=False allows us to apply StandardScaler explicitly afterwards.
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False)

        if Config.APPLY_SCALING:
            self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the pipeline to the training data.

        Args:
            X: Input data (DataFrame or numpy array).
        """
        # Enforce float64 precision
        X_64 = np.array(X, dtype=Config.FLOAT_PRECISION)

        # 1. Fit PowerTransformer
        if self.pt:
            self.pt.fit(X_64)
            # Transform temporarily to get the distribution for the Scaler
            X_trans = self.pt.transform(X_64)
        else:
            X_trans = X_64

        # 2. Fit StandardScaler
        if self.scaler:
            self.scaler.fit(X_trans)

        return self

    def transform(self, X):
        """
        Apply the learned transformations to the data.

        Args:
            X: Input data (DataFrame or numpy array).

        Returns:
            Transformed data as a float64 numpy array.
        """
        # Enforce float64 precision
        X_64 = np.array(X, dtype=Config.FLOAT_PRECISION)

        # 1. Apply PowerTransformer
        if self.pt:
            X_trans = self.pt.transform(X_64)
        else:
            X_trans = X_64

        # 2. Apply StandardScaler
        if self.scaler:
            X_trans = self.scaler.transform(X_trans)

        return X_trans.astype(Config.FLOAT_PRECISION)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        self.fit(X)
        return self.transform(X)


def load_and_preprocess_data(load_cached_data=True):
    """
    Orchestrates the data loading, cleaning, preprocessing, and caching.

    1. Loads raw/extracted features via library.feature_extraction.
    2. Removes excluded features defined in Config.
    3. Fits HighPrecisionPipeline on Training data.
    4. Transforms Train, Val, and Test data.
    5. Caches the resulting numpy arrays to disk.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed arrays from cache.

    Returns:
        Tuple containing:
        (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test)
    """
    # Define cache file paths
    cache_files = {
        "X_train": "X_train_processed.npy",
        "y_train": "y_train_processed.npy",
        "ids_train": "ids_train_processed.npy",
        "X_val": "X_val_processed.npy",
        "y_val": "y_val_processed.npy",
        "ids_val": "ids_val_processed.npy",
        "X_test": "X_test_processed.npy",
        "ids_test": "ids_test_processed.npy",
    }

    # Check if cache exists
    cache_exists = True
    for fname in cache_files.values():
        fpath = os.path.join(Config.CACHE_DIR, fname)
        if not os.path.exists(fpath):
            cache_exists = False
            break

    if load_cached_data and cache_exists:
        print(f"Loading preprocessed data from cache: {Config.CACHE_DIR}")
        try:
            data = {}
            for key, fname in cache_files.items():
                fpath = os.path.join(Config.CACHE_DIR, fname)
                data[key] = np.load(fpath, allow_pickle=True)

            return (
                data["X_train"],
                data["y_train"],
                data["ids_train"],
                data["X_val"],
                data["y_val"],
                data["ids_val"],
                data["X_test"],
                data["ids_test"],
            )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print("Preprocessing data from scratch...")

    # 1. Load Data (Features + Targets)
    # feature_extraction handles the extraction cache
    df_train, y_train, ids_train = fe.get_train_data(load_cached_data=load_cached_data)
    df_val, y_val, ids_val = fe.get_val_data(load_cached_data=load_cached_data)
    df_test, ids_test = fe.get_test_data(load_cached_data=load_cached_data)

    # 2. Clean Data (Remove Excluded Features)
    def clean_df(df):
        # Identify columns to drop
        to_drop = [c for c in Config.EXCLUDED_FEATURES if c in df.columns]
        if to_drop:
            return df.drop(columns=to_drop)
        return df

    X_train_clean = clean_df(df_train)
    X_val_clean = clean_df(df_val)
    X_test_clean = clean_df(df_test)

    # 3. Fit Pipeline (Train only)
    pipeline = HighPrecisionPipeline()
    print("Fitting HighPrecisionPipeline on training data...")
    X_train_trans = pipeline.fit_transform(X_train_clean)

    # 4. Transform Val and Test
    print("Transforming validation and test data...")
    X_val_trans = pipeline.transform(X_val_clean)
    X_test_trans = pipeline.transform(X_test_clean)

    # 5. Save to Cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    print(f"Saving preprocessed data to cache: {Config.CACHE_DIR}")

    np.save(os.path.join(Config.CACHE_DIR, cache_files["X_train"]), X_train_trans)
    np.save(os.path.join(Config.CACHE_DIR, cache_files["y_train"]), y_train)
    np.save(os.path.join(Config.CACHE_DIR, cache_files["ids_train"]), ids_train)

    np.save(os.path.join(Config.CACHE_DIR, cache_files["X_val"]), X_val_trans)
    np.save(os.path.join(Config.CACHE_DIR, cache_files["y_val"]), y_val)
    np.save(os.path.join(Config.CACHE_DIR, cache_files["ids_val"]), ids_val)

    np.save(os.path.join(Config.CACHE_DIR, cache_files["X_test"]), X_test_trans)
    np.save(os.path.join(Config.CACHE_DIR, cache_files["ids_test"]), ids_test)

    return (
        X_train_trans,
        y_train,
        ids_train,
        X_val_trans,
        y_val,
        ids_val,
        X_test_trans,
        ids_test,
    )
