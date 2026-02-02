import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.utils import set_seed

# Constants
CACHE_DIR = "./working/idea_5/"
METADATA_DIR = "./metadata"


def load_full_dataset(load_cached_data=True):
    """
    Loads the dataset, concatenating train and validation sets for maximum sample utilization.
    Handles caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (X_train, y_train, X_test, test_ids, classes)
            - X_train (np.ndarray): Feature matrix for training (combined train + val).
            - y_train (np.ndarray): Encoded target vector for training.
            - X_test (np.ndarray): Feature matrix for testing.
            - test_ids (np.ndarray): IDs for test samples.
            - classes (np.ndarray): Array of original class names corresponding to encoded integers.
    """
    set_seed(42)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # 1. Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"], allow_pickle=True)
            return X_train, y_train, X_test, test_ids, classes
        else:
            print("Cache not found or incomplete. Processing from scratch...")

    # 2. Process from scratch
    print("Loading metadata CSVs...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"One or more metadata files missing in {METADATA_DIR}")

    df_train_part = pd.read_csv(train_path)
    df_val_part = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Concatenate train and val sets
    print("Concatenating training and validation sets...")
    df_train_full = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Identify feature columns (exclude metadata)
    # Metadata columns usually: id, species, image_path
    # We want margin_*, shape_*, texture_*
    exclude_cols = {"id", "species", "image_path"}
    feature_cols = [c for c in df_train_full.columns if c not in exclude_cols]

    # Ensure consistent column ordering
    feature_cols.sort()

    print(f"Selected {len(feature_cols)} features.")

    # Prepare X matrices
    X_train = df_train_full[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Prepare targets
    print("Encoding targets...")
    le = LabelEncoder()
    y_train = le.fit_transform(df_train_full["species"])
    classes = le.classes_

    # Prepare IDs
    test_ids = df_test["id"].values

    # 3. Save to cache
    print(f"Saving processed data to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_test, test_ids, classes
