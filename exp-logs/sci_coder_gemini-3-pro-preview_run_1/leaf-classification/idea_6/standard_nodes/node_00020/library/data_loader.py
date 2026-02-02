import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import Config


def load_dataset(load_cached_data=True):
    """
    Loads the leaf classification dataset.

    Logic:
    1. Checks if cached .npy files exist in Config.WORKING_DIR.
    2. If load_cached_data is True and files exist, loads and returns them.
    3. Otherwise, reads metadata CSVs, processes features/labels, caches the results, and returns them.

    Returns:
        X_train (np.ndarray): Training features (N_train, 192).
        y_train (np.ndarray): Training labels (N_train,).
        X_val (np.ndarray): Validation features (N_val, 192).
        y_val (np.ndarray): Validation labels (N_val,).
        X_test (np.ndarray): Test features (N_test, 192).
        test_ids (np.ndarray): Test image IDs (N_test,).
        classes (np.ndarray): Array of class names (strings).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filenames
    # Append _debug to filenames if in debug mode to avoid corrupting full cache
    suffix = "_debug.npy" if Config.DEBUG else ".npy"

    path_X_train = os.path.join(Config.WORKING_DIR, f"loaded_X_train{suffix}")
    path_y_train = os.path.join(Config.WORKING_DIR, f"loaded_y_train{suffix}")
    path_X_val = os.path.join(Config.WORKING_DIR, f"loaded_X_val{suffix}")
    path_y_val = os.path.join(Config.WORKING_DIR, f"loaded_y_val{suffix}")
    path_X_test = os.path.join(Config.WORKING_DIR, f"loaded_X_test{suffix}")
    path_ids = os.path.join(Config.WORKING_DIR, f"loaded_test_ids{suffix}")
    path_classes = os.path.join(Config.WORKING_DIR, f"loaded_classes{suffix}")

    # 1. Try to load from cache
    if load_cached_data:
        try:
            if (
                os.path.exists(path_X_train)
                and os.path.exists(path_y_train)
                and os.path.exists(path_X_val)
                and os.path.exists(path_y_val)
                and os.path.exists(path_X_test)
                and os.path.exists(path_ids)
                and os.path.exists(path_classes)
            ):

                print(f"Loading cached dataset from {Config.WORKING_DIR}...")
                X_train = np.load(path_X_train)
                y_train = np.load(path_y_train)
                X_val = np.load(path_X_val)
                y_val = np.load(path_y_val)
                X_test = np.load(path_X_test)
                test_ids = np.load(path_ids)
                classes = np.load(path_classes, allow_pickle=True)

                return X_train, y_train, X_val, y_val, X_test, test_ids, classes
            else:
                print("Cache miss. Processing data from scratch...")
        except Exception as e:
            print(f"Error loading cache: {e}. Reprocessing data...")
    else:
        print("Force reload. Processing data from scratch...")

    # 2. Process from scratch
    print("Reading metadata CSVs...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Slicing datasets to {Config.DEBUG_SAMPLES} samples.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLES]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLES]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLES]

    # Identify Feature Columns
    # Exclude non-feature columns. The metadata contains 'id', 'species', 'file_path'.
    # We want margin_*, shape_*, texture_*
    exclude_cols = {"id", "species", "file_path"}
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Verify feature count
    if len(feature_cols) != Config.N_FEATURES:
        print(
            f"Warning: Expected {Config.N_FEATURES} features, found {len(feature_cols)}."
        )

    # Extract Features
    print("Extracting features...")
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Extract Targets and Encode
    print("Encoding targets...")
    le = LabelEncoder()
    # Fit on training species. Stratified split ensures all classes are present.
    y_train = le.fit_transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    classes = le.classes_

    # Extract Test IDs
    test_ids = df_test["id"].values

    # 3. Save to cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    np.save(path_X_train, X_train)
    np.save(path_y_train, y_train)
    np.save(path_X_val, X_val)
    np.save(path_y_val, y_val)
    np.save(path_X_test, X_test)
    np.save(path_ids, test_ids)
    np.save(path_classes, classes)

    print("Data loading complete.")
    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
