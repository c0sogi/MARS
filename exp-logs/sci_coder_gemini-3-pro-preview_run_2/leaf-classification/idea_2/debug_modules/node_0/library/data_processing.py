import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


def load_and_preprocess_data(load_cached_data=True):
    """
    Loads, preprocesses, and caches the Leaf Classification dataset.

    Steps:
    1. Checks for cached .npy files in Config.CACHE_DIR.
    2. If found and load_cached_data is True, returns cached data.
    3. If not, loads raw CSVs from metadata.
    4. Extracts features based on Config.FEATURE_GROUPS.
    5. Encodes target labels (species) using LabelEncoder.
    6. Scales features using StandardScaler (fit on train, transform val/test).
    7. Caches the processed arrays to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """

    # Define cache file paths
    cache_dir = Config.CACHE_DIR
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # 1. Try Loading from Cache
    if load_cached_data:
        # Check if all required files exist
        if all(os.path.exists(p) for p in files.values()):
            print(f"Loading processed data from cache: {cache_dir}")
            try:
                X_train = np.load(files["X_train"])
                y_train = np.load(files["y_train"])
                X_val = np.load(files["X_val"])
                y_val = np.load(files["y_val"])
                X_test = np.load(files["X_test"])
                test_ids = np.load(files["test_ids"])
                classes = np.load(files["classes"], allow_pickle=True)

                return X_train, y_train, X_val, y_val, X_test, test_ids, classes
            except Exception as e:
                print(f"Error loading cache: {e}. Reprocessing data...")
        else:
            print("Cache incomplete or missing. Reprocessing data...")
    else:
        print("Force reloading data. Reprocessing...")

    # 2. Load Raw Metadata
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 3. Feature Selection
    # Identify columns belonging to the specified feature groups (margin, shape, texture)
    all_columns = df_train.columns
    feature_cols = []

    for group in Config.FEATURE_GROUPS:
        # Select columns that contain the group name (e.g., 'margin_1')
        group_cols = [c for c in all_columns if group in c]
        feature_cols.extend(group_cols)

    # Sort columns to ensure deterministic order across runs
    feature_cols = sorted(list(set(feature_cols)))

    print(
        f"Selected {len(feature_cols)} features based on groups: {Config.FEATURE_GROUPS}"
    )

    # Extract raw feature matrices
    X_train_raw = df_train[feature_cols].values
    X_val_raw = df_val[feature_cols].values
    X_test_raw = df_test[feature_cols].values

    # Extract targets and IDs
    y_train_str = df_train["species"].values
    y_val_str = df_val["species"].values
    test_ids = df_test["id"].values

    # 4. Target Encoding
    print("Encoding target labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    # Transform validation set (assuming stratified split ensures all classes are in train)
    y_val = le.transform(y_val_str)
    classes = le.classes_

    # 5. Feature Scaling
    print(f"Scaling features using {Config.SCALER_TYPE}...")
    if Config.SCALER_TYPE == "standard":
        scaler = StandardScaler()
        # Fit on training data only to prevent data leakage
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        # Fallback or no scaling (though Config defaults to standard)
        X_train = X_train_raw
        X_val = X_val_raw
        X_test = X_test_raw

    # 6. Save to Cache
    print(f"Saving processed data to cache: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["X_val"], X_val)
    np.save(files["y_val"], y_val)
    np.save(files["X_test"], X_test)
    np.save(files["test_ids"], test_ids)
    np.save(files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
