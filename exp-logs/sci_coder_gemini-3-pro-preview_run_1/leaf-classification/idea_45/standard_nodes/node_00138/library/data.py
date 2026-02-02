import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
import library.config as config
import library.utils as utils
import library.features as features


def get_hashed_cache_paths():
    """
    Generates cache file paths that include the configuration hash.
    This ensures that changes in feature configuration or preprocessing
    trigger a re-computation of the dataset.
    """
    config_hash = utils.generate_config_hash()

    # Construct paths with the hash in the filename
    train_path = os.path.join(
        config.WORKING_DIR, f"train_features_{config_hash}.parquet"
    )
    val_path = os.path.join(config.WORKING_DIR, f"val_features_{config_hash}.parquet")
    test_path = os.path.join(config.WORKING_DIR, f"test_features_{config_hash}.parquet")

    return train_path, val_path, test_path


def load_data(debug_sample_size=None, load_cached_data=True):
    """
    Loads, preprocesses, and splits the data for the Moment-Completed High-Precision OAS Discriminant.

    Steps:
    1. Resolve cache paths based on configuration hash.
    2. Load (or compute) Train, Validation, and Test datasets using library.features.
    3. Enforce alphanumeric feature ordering.
    4. Apply Inductive Preprocessing (Yeo-Johnson + StandardScaling fitted on Train).
    5. Encode targets.
    6. Return float64 arrays for high-precision inference.

    Args:
        debug_sample_size (int, optional): If set, truncates datasets for rapid debugging.
        load_cached_data (bool): Whether to attempt loading from disk cache.

    Returns:
        dict: Contains X_train, y_train, X_val, y_val, X_test, test_ids, label_encoder, and feature_names.
    """
    # 1. Resolve Paths
    train_cache_path, val_cache_path, test_cache_path = get_hashed_cache_paths()

    print(f"Loading data with config hash: {utils.generate_config_hash()}")

    # 2. Load Dataframes
    # library.features.process_dataset handles the check-cache -> compute -> save logic
    df_train = features.process_dataset(
        config.TRAIN_METADATA_PATH, train_cache_path, load_cached_data
    )
    df_val = features.process_dataset(
        config.VAL_METADATA_PATH, val_cache_path, load_cached_data
    )
    df_test = features.process_dataset(
        config.TEST_METADATA_PATH, test_cache_path, load_cached_data
    )

    # 3. Debug Sampling
    if debug_sample_size is not None:
        print(f"DEBUG: Subsampling data to {debug_sample_size} rows.")
        df_train = df_train.iloc[:debug_sample_size]
        df_val = df_val.iloc[:debug_sample_size]
        # We generally keep test full for submission generation, but can sample if strictly needed
        # For this implementation, we'll leave test intact unless specifically requested,
        # but to be consistent with a 'debug run', we can sample it too.
        df_test = df_test.iloc[:debug_sample_size]

    # 4. Feature Selection & Ordering
    # Combine tabular and image features
    feature_cols = config.TABULAR_FEATURE_COLS + config.IMAGE_FEATURE_COLS

    # Enforce Alphanumeric Column Ordering for deterministic memory layout
    feature_cols = sorted(feature_cols)

    print(f"Total Features: {len(feature_cols)}")

    # 5. Extract Raw Arrays (float64)
    X_train_raw = df_train[feature_cols].values.astype(config.FLOAT_PRECISION)
    y_train_raw = df_train[config.TARGET_COL].values

    X_val_raw = df_val[feature_cols].values.astype(config.FLOAT_PRECISION)
    y_val_raw = df_val[config.TARGET_COL].values

    X_test_raw = df_test[feature_cols].values.astype(config.FLOAT_PRECISION)
    test_ids = df_test[config.ID_COL].values

    # 6. Inductive Preprocessing
    # "Transformation is applied to all features... Fitted only on the Training set"
    print("Applying Yeo-Johnson transformation and Standard Scaling...")

    # Initialize Transformers
    # Yeo-Johnson stabilizes variance for geometric features
    pt = PowerTransformer(method=config.PREPROCESS_POWER_METHOD, standardize=False)
    scaler = StandardScaler()

    # Fit on Train ONLY
    pt.fit(X_train_raw)
    X_train_pt = pt.transform(X_train_raw)

    scaler.fit(X_train_pt)
    X_train_final = scaler.transform(X_train_pt)

    # Transform Val
    X_val_pt = pt.transform(X_val_raw)
    X_val_final = scaler.transform(X_val_pt)

    # Transform Test
    X_test_pt = pt.transform(X_test_raw)
    X_test_final = scaler.transform(X_test_pt)

    # 7. Label Encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)

    # Transform validation labels.
    # Note: In a real scenario, we handle unseen labels, but this dataset is closed-set stratified.
    y_val_enc = le.transform(y_val_raw)

    print("Data processing complete.")
    print(
        f"Train shape: {X_train_final.shape}, Val shape: {X_val_final.shape}, Test shape: {X_test_final.shape}"
    )

    return {
        "X_train": X_train_final,
        "y_train": y_train_enc,
        "X_val": X_val_final,
        "y_val": y_val_enc,
        "X_test": X_test_final,
        "test_ids": test_ids,
        "label_encoder": le,
        "feature_names": feature_cols,
    }
