import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library import config, utils


class LeafPipeline:
    """
    Handles data loading, feature extraction, and the inductive preprocessing pipeline.
    Ensures transformations are fitted only on training data and applied consistently.
    """

    def __init__(self):
        # Initialize transformers based on configuration
        # Yeo-Johnson is used to normalize feature distributions
        self.power_transformer = (
            PowerTransformer(method="yeo-johnson", standardize=False)
            if config.APPLY_POWER_TRANSFORM
            else None
        )
        # Standard Scaling (z-score)
        self.scaler = StandardScaler() if config.APPLY_SCALING else None
        # Label Encoder for target variable
        self.label_encoder = LabelEncoder()

        # Load column definitions from config
        self.feature_cols = config.FEATURE_COLS
        self.target_col = config.TARGET_COL
        self.id_col = config.ID_COL

    def _load_raw_df(self, path):
        """Loads a CSV file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path)

    def fit_transform_train(self, df):
        """
        Fits the pipeline on the training data and transforms it.

        Args:
            df (pd.DataFrame): Raw training dataframe.

        Returns:
            tuple: (X_transformed, y_encoded, ids)
        """
        # Extract features and enforce float64 precision
        X = utils.enforce_float64(df[self.feature_cols].values)
        y = df[self.target_col].values
        ids = df[self.id_col].values

        # 1. Fit and Transform PowerTransformer (if enabled)
        if self.power_transformer:
            X = self.power_transformer.fit_transform(X)

        # 2. Fit and Transform StandardScaler (if enabled)
        if self.scaler:
            X = self.scaler.fit_transform(X)

        # 3. Fit and Transform LabelEncoder
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded, ids

    def transform(self, df, is_test=False):
        """
        Transforms validation or test data using the fitted pipeline.

        Args:
            df (pd.DataFrame): Raw dataframe.
            is_test (bool): If True, skips target processing.

        Returns:
            tuple: (X_transformed, y_encoded_or_None, ids)
        """
        X = utils.enforce_float64(df[self.feature_cols].values)
        ids = df[self.id_col].values

        # 1. Transform PowerTransformer
        if self.power_transformer:
            X = self.power_transformer.transform(X)

        # 2. Transform StandardScaler
        if self.scaler:
            X = self.scaler.transform(X)

        if is_test:
            return X, None, ids
        else:
            y = df[self.target_col].values
            # Transform targets using the encoder fitted on train
            y_encoded = self.label_encoder.transform(y)
            return X, y_encoded, ids


def get_processed_data(load_cached_data=True):
    """
    Main entry point to get processed training, validation, and test data.
    Implements caching to avoid re-processing.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test, class_labels)
    """
    # Define cache filenames
    cache_files = {
        "X_train": "X_train.npy",
        "y_train": "y_train.npy",
        "ids_train": "ids_train.npy",
        "X_val": "X_val.npy",
        "y_val": "y_val.npy",
        "ids_val": "ids_val.npy",
        "X_test": "X_test.npy",
        "ids_test": "ids_test.npy",
        "classes": "classes.npy",
    }

    # Check if all cache files exist
    cache_exists = all(
        os.path.exists(os.path.join(config.WORKING_DIR, f))
        for f in cache_files.values()
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        data = {}
        for key, filename in cache_files.items():
            path = os.path.join(config.WORKING_DIR, filename)
            # allow_pickle=True is required for string arrays (classes)
            data[key] = np.load(path, allow_pickle=True)

        return (
            data["X_train"],
            data["y_train"],
            data["ids_train"],
            data["X_val"],
            data["y_val"],
            data["ids_val"],
            data["X_test"],
            data["ids_test"],
            data["classes"],
        )

    print("Processing data from scratch...")
    pipeline = LeafPipeline()

    # Load Raw Data using metadata paths
    df_train = pipeline._load_raw_df(config.TRAIN_DATA_PATH)
    df_val = pipeline._load_raw_df(config.VAL_DATA_PATH)
    df_test = pipeline._load_raw_df(config.TEST_DATA_PATH)

    # Process Train (Fit + Transform)
    X_train, y_train, ids_train = pipeline.fit_transform_train(df_train)

    # Process Val (Transform only)
    X_val, y_val, ids_val = pipeline.transform(df_val, is_test=False)

    # Process Test (Transform only)
    X_test, _, ids_test = pipeline.transform(df_test, is_test=True)

    # Get Class Labels (mapping from int back to string)
    classes = pipeline.label_encoder.classes_

    # Save to Cache
    data_map = {
        "X_train": X_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_val": X_val,
        "y_val": y_val,
        "ids_val": ids_val,
        "X_test": X_test,
        "ids_test": ids_test,
        "classes": classes,
    }

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    for key, val in data_map.items():
        filename = cache_files[key]
        path = os.path.join(config.WORKING_DIR, filename)
        np.save(path, val)

    print(f"Data processed and saved to {config.WORKING_DIR}")

    return (
        X_train,
        y_train,
        ids_train,
        X_val,
        y_val,
        ids_val,
        X_test,
        ids_test,
        classes,
    )
