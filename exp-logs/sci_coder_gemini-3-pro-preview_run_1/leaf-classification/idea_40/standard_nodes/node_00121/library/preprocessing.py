import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.config import WORKING_DIR, FLOAT_PRECISION, SEED
from library.features import get_dataset
from library.utils import get_config_hash


class HighPrecisionPreprocessor:
    """
    A wrapper class for the high-precision preprocessing pipeline.
    It applies Yeo-Johnson transformation followed by Standard Scaling,
    strictly maintaining float64 precision.
    """

    def __init__(self):
        # standardize=False for PowerTransformer because we apply StandardScaler afterwards
        # This allows us to control the scaling step explicitly
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the training data.

        Args:
            X (np.ndarray): Training features.
        """
        # Ensure float64
        X_flt = X.astype(FLOAT_PRECISION)

        # Fit PowerTransformer
        self.pt.fit(X_flt)

        # Transform to get intermediate state for scaler fitting
        X_pt = self.pt.transform(X_flt)

        # Fit StandardScaler
        self.scaler.fit(X_pt)

        return self

    def transform(self, X):
        """
        Transforms the data using the fitted transformers.

        Args:
            X (np.ndarray): Features to transform.

        Returns:
            np.ndarray: Transformed features in float64.
        """
        X_flt = X.astype(FLOAT_PRECISION)
        X_pt = self.pt.transform(X_flt)
        X_scaled = self.scaler.transform(X_pt)
        return X_scaled.astype(FLOAT_PRECISION)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def get_preprocessed_data(load_cached_data=True, max_samples=None):
    """
    Loads raw features, aligns columns, fits the preprocessor on training data,
    transforms all splits, and returns numpy arrays ready for modeling.

    Implements caching based on the configuration hash.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): Maximum number of samples to load for debugging.
                                     If set, caching is bypassed to prevent corruption.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Generate hash for cache filenames
    config_hash = get_config_hash()

    # Define cache paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, f"X_train_{config_hash}.npy"),
        "y_train": os.path.join(WORKING_DIR, f"y_train_{config_hash}.npy"),
        "X_val": os.path.join(WORKING_DIR, f"X_val_{config_hash}.npy"),
        "y_val": os.path.join(WORKING_DIR, f"y_val_{config_hash}.npy"),
        "X_test": os.path.join(WORKING_DIR, f"X_test_{config_hash}.npy"),
        "test_ids": os.path.join(WORKING_DIR, f"test_ids_{config_hash}.npy"),
        "classes": os.path.join(WORKING_DIR, f"classes_{config_hash}.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(path) for path in cache_files.values())

    # Logic: If requesting full data and cache exists, load it.
    if load_cached_data and all_cached:
        print("Loading preprocessed data from cache...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"])
        classes = np.load(cache_files["classes"], allow_pickle=True)

        if max_samples is not None:
            print(f"Slicing data to {max_samples} samples...")
            X_train = X_train[:max_samples]
            y_train = y_train[:max_samples]
            X_val = X_val[:max_samples]
            y_val = y_val[:max_samples]
            X_test = X_test[:max_samples]
            test_ids = test_ids[:max_samples]

        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Preprocessing data from scratch...")

    # 1. Load Dataframes with Features
    # Note: get_dataset handles the feature extraction caching internally
    df_train = get_dataset("train", load_cached_data=load_cached_data)
    df_val = get_dataset("val", load_cached_data=load_cached_data)
    df_test = get_dataset("test", load_cached_data=load_cached_data)

    # Apply max_samples if set (before processing to save time)
    if max_samples is not None:
        print(f"Limiting raw data to {max_samples} samples (Debug Mode)...")

        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows
        # Naive slicing can lead to disjoint label sets in train/val.
        # We select a subset of classes to ensure overlap and density.
        unique_species = df_train["species"].unique()
        # Select first 10 classes (or fewer if not enough) to ensure we have enough samples per class
        target_classes = unique_species[:10]

        # Filter DataFrames to these classes
        df_train = df_train[df_train["species"].isin(target_classes)]
        df_val = df_val[df_val["species"].isin(target_classes)]

        # Apply the hard sample limit
        df_train = df_train.iloc[:max_samples]
        df_val = df_val.iloc[:max_samples]
        df_test = df_test.iloc[:max_samples]

        # Ensure validation set only contains labels present in the training set
        # This prevents LabelEncoder from seeing unseen labels
        final_train_species = df_train["species"].unique()
        df_val = df_val[df_val["species"].isin(final_train_species)]

    # 2. Identify Feature Columns
    # Exclude non-feature columns
    exclude_cols = ["id", "species", "file_path"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Enforce Alphanumeric Column Ordering for determinism
    feature_cols.sort()

    print(f"Number of features: {len(feature_cols)}")

    # 3. Extract Arrays
    X_train_raw = df_train[feature_cols].values.astype(FLOAT_PRECISION)
    y_train_raw = df_train["species"].values

    X_val_raw = df_val[feature_cols].values.astype(FLOAT_PRECISION)
    y_val_raw = df_val["species"].values

    X_test_raw = df_test[feature_cols].values.astype(FLOAT_PRECISION)
    test_ids = df_test["id"].values

    # 4. Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    # Handle unseen labels in val if any (though stratified split should prevent this)
    y_val_enc = le.transform(y_val_raw)
    classes = le.classes_

    # 5. Apply Preprocessing Pipeline
    # Inductive Fit: Fit only on Train
    preprocessor = HighPrecisionPreprocessor()
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train_raw)

    print("Transforming datasets...")
    X_train_trans = preprocessor.transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # 6. Save to Cache (Only if full dataset was processed)
    if max_samples is None:
        print("Saving preprocessed data to cache...")
        np.save(cache_files["X_train"], X_train_trans)
        np.save(cache_files["y_train"], y_train_enc)
        np.save(cache_files["X_val"], X_val_trans)
        np.save(cache_files["y_val"], y_val_enc)
        np.save(cache_files["X_test"], X_test_trans)
        np.save(cache_files["test_ids"], test_ids)
        np.save(cache_files["classes"], classes)
    else:
        print("Skipping cache save due to max_samples limit.")

    return (
        X_train_trans,
        y_train_enc,
        X_val_trans,
        y_val_enc,
        X_test_trans,
        test_ids,
        classes,
    )
