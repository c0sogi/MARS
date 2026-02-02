import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.utils import load_metadata

# Set random seed for reproducibility
np.random.seed(42)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, separating features and targets, and encoding labels.
    Implements caching to ./working/idea_9/ to speed up subsequent loads.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, class_names)
    """
    cache_dir = "./working/idea_9/"
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"])
        class_names = np.load(cache_files["classes"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_ids, class_names

    print("Loading data from metadata files...")
    # Load metadata using the provided utility
    df_train = load_metadata("train")
    df_val = load_metadata("val")
    df_test = load_metadata("test")

    # Identify feature columns
    # We exclude non-feature columns. Note: test set doesn't have 'species'.
    exclude_cols = ["id", "species", "image_path"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Extract Features
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Extract Targets and Encode
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    class_names = le.classes_

    # Extract Test IDs
    test_ids = df_test["id"].values

    # Save to cache
    print("Saving data to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], class_names)

    return X_train, y_train, X_val, y_val, X_test, test_ids, class_names
