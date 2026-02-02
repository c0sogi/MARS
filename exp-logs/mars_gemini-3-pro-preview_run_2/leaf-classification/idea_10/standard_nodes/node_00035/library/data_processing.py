import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    FEATURE_VIEWS,
    RANDOM_SEED,
)


def load_and_combine_data(debug_size=None):
    """
    Loads train, validation, and test data from metadata CSVs.
    Combines train and validation sets into a single training set.

    Args:
        debug_size (int, optional): Number of rows to limit the training data to for debugging.

    Returns:
        tuple: (df_full_train, df_test)
    """
    # Load Metadata
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Combine Train and Validation for maximum data utilization
    df_full_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    # Debugging: Subset data if requested
    if debug_size is not None and debug_size < len(df_full_train):
        df_full_train = df_full_train.iloc[:debug_size]

    return df_full_train, df_test


def extract_views(df):
    """
    Extracts Global, Margin, Shape, and Texture views from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing feature columns.

    Returns:
        dict: Dictionary mapping view names to numpy arrays of features.
    """
    # Identify feature columns (exclude metadata)
    exclude_cols = ["id", "image_path", "species"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    views = {}

    # 1. Global View (All features)
    views["Global"] = df[feature_cols].values.astype(np.float32)

    # 2. Specific Views based on config
    # FEATURE_VIEWS = {"Margin": "margin", "Shape": "shape", "Texture": "texture"}
    for view_name, prefix in FEATURE_VIEWS.items():
        # Select columns starting with the prefix
        view_cols = [c for c in feature_cols if c.startswith(prefix)]
        if not view_cols:
            continue
        views[view_name] = df[view_cols].values.astype(np.float32)

    return views


def scale_views(train_views, test_views):
    """
    Applies StandardScaler to each view independently.
    Fits on the training set, transforms both training and test sets.

    Args:
        train_views (dict): Dictionary of training feature arrays.
        test_views (dict): Dictionary of test feature arrays.

    Returns:
        tuple: (scaled_train_views, scaled_test_views)
    """
    scaled_train_views = {}
    scaled_test_views = {}

    for view_name in train_views.keys():
        if view_name not in test_views:
            continue

        scaler = StandardScaler()

        # Fit on training data only to prevent leakage
        scaled_train_views[view_name] = scaler.fit_transform(train_views[view_name])

        # Transform test data
        scaled_test_views[view_name] = scaler.transform(test_views[view_name])

    return scaled_train_views, scaled_test_views


def process_data(load_cached_data=True, debug_size=None):
    """
    Main data processing function.
    Loads, combines, extracts views, scales, and encodes labels.
    Implements caching mechanism to avoid re-processing.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.
        debug_size (int, optional): Limit training data size.

    Returns:
        tuple: (X_train_views, y_train, X_test_views, test_ids, classes)
    """
    # Define cache file paths
    # Using .npz for dictionaries of arrays and .npy for single arrays
    cache_train_path = os.path.join(WORKING_DIR, "X_train_views.npz")
    cache_test_path = os.path.join(WORKING_DIR, "X_test_views.npz")
    cache_y_path = os.path.join(WORKING_DIR, "y_train.npy")
    cache_ids_path = os.path.join(WORKING_DIR, "test_ids.npy")
    cache_classes_path = os.path.join(WORKING_DIR, "classes.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_test_path)
            and os.path.exists(cache_y_path)
            and os.path.exists(cache_ids_path)
            and os.path.exists(cache_classes_path)
        ):

            # Load npz files
            train_npz = np.load(cache_train_path)
            test_npz = np.load(cache_test_path)

            # Reconstruct dictionaries from npz files
            X_train_views = {k: train_npz[k] for k in train_npz.files}
            X_test_views = {k: test_npz[k] for k in test_npz.files}

            y_train = np.load(cache_y_path)
            test_ids = np.load(cache_ids_path)
            classes = np.load(cache_classes_path)

            return X_train_views, y_train, X_test_views, test_ids, classes

    # 2. Process from Scratch

    # Load and Combine Data
    df_train_full, df_test = load_and_combine_data(debug_size=debug_size)

    # Extract IDs and Targets
    y_raw = df_train_full["species"].values
    test_ids = df_test["id"].values

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)
    classes = le.classes_

    # Extract Views (Raw Features)
    X_train_views_raw = extract_views(df_train_full)
    X_test_views_raw = extract_views(df_test)

    # Scale Views
    X_train_views, X_test_views = scale_views(X_train_views_raw, X_test_views_raw)

    # 3. Save to Cache
    os.makedirs(WORKING_DIR, exist_ok=True)

    np.savez(cache_train_path, **X_train_views)
    np.savez(cache_test_path, **X_test_views)
    np.save(cache_y_path, y_train)
    np.save(cache_ids_path, test_ids)
    np.save(cache_classes_path, classes)

    return X_train_views, y_train, X_test_views, test_ids, classes
