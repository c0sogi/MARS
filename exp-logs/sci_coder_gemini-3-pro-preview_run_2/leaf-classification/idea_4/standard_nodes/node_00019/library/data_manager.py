import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    CACHE_DIR,
    ID_COL,
    TARGET_COL,
)


def load_and_preprocess_data(load_cached_data=True):
    """
    Loads data from metadata CSVs, merges train and validation sets,
    performs scaling and encoding, and returns processed numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
                                 If False or cache missing, reprocesses data.

    Returns:
        X_train (np.ndarray): Scaled training features (Train + Val).
        y_train (np.ndarray): Encoded training labels.
        X_test (np.ndarray): Scaled test features.
        test_ids (np.ndarray): IDs for the test set.
        classes (np.ndarray): List of class names corresponding to encoded labels.
    """

    # Define cache paths
    cache_paths = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_paths.values()):
            print("Loading processed data from cache...")
            try:
                X_train = np.load(cache_paths["X_train"])
                y_train = np.load(cache_paths["y_train"])
                X_test = np.load(cache_paths["X_test"])
                test_ids = np.load(cache_paths["test_ids"])
                classes = np.load(cache_paths["classes"], allow_pickle=True)
                return X_train, y_train, X_test, test_ids, classes
            except Exception as e:
                print(f"Error loading cache: {e}. Reprocessing...")
        else:
            print("Cache missing. Reprocessing...")
    else:
        print("Ignoring cache. Reprocessing...")

    # Load metadata
    print("Loading metadata CSVs...")
    if (
        not os.path.exists(TRAIN_FILE)
        or not os.path.exists(VAL_FILE)
        or not os.path.exists(TEST_FILE)
    ):
        raise FileNotFoundError(
            f"Metadata files missing. Expected at: {TRAIN_FILE}, {VAL_FILE}, {TEST_FILE}"
        )

    df_train_part = pd.read_csv(TRAIN_FILE)
    df_val_part = pd.read_csv(VAL_FILE)
    df_test = pd.read_csv(TEST_FILE)

    # Merge Train and Validation for full training
    # The strategy requires training on the full dataset (Train + Val)
    df_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Identify features
    # Exclude non-feature columns. Note: 'image_path' is in metadata but not used for tabular models.
    non_feature_cols = {ID_COL, TARGET_COL, "image_path"}
    feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

    print(f"Identified {len(feature_cols)} feature columns.")

    # Extract raw data
    X_train_raw = df_train[feature_cols].values
    y_train_raw = df_train[TARGET_COL].values

    # Ensure test set uses same feature order
    # Test csv does not have TARGET_COL, but has ID and image_path
    X_test_raw = df_test[feature_cols].values
    test_ids = df_test[ID_COL].values

    # Encode Targets
    print("Encoding targets...")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    classes = le.classes_

    # Scale Features
    # Global scaling using StandardScaler as per strategy
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Save to cache
    print(f"Saving processed data to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_paths["X_train"], X_train)
    np.save(cache_paths["y_train"], y_train)
    np.save(cache_paths["X_test"], X_test)
    np.save(cache_paths["test_ids"], test_ids)
    np.save(cache_paths["classes"], classes)

    print("Data processing complete.")
    return X_train, y_train, X_test, test_ids, classes
