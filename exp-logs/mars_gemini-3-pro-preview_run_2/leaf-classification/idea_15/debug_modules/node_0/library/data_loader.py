import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define constants
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_15"


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, concatenating train and validation sets for maximum sample size.
    Extracts numerical features (margin, shape, texture) and encodes targets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (X_train, y_train, X_test, test_ids, label_encoder)
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # Check if we should and can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading cached data from {CACHE_DIR}...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"], allow_pickle=True)

            # Reconstruct LabelEncoder
            le = LabelEncoder()
            le.classes_ = classes

            return X_train, y_train, X_test, test_ids, le
        else:
            print("Cache miss or partial cache found. Re-processing data...")

    # Load metadata
    print("Loading metadata CSVs...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    df_train_part = pd.read_csv(train_path)
    df_val_part = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Concatenate train and val sets as per strategy (maximize sample size)
    df_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Identify feature columns (margin, shape, texture)
    # We exclude id, species, image_path
    feature_cols = [
        c for c in df_train.columns if c not in ["id", "species", "image_path"]
    ]

    # Sort columns to ensure consistency
    feature_cols.sort()

    print(f"Extracting {len(feature_cols)} features...")

    # Extract features
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Extract IDs
    test_ids = df_test["id"].values

    # Encode Targets
    print("Encoding targets...")
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["species"])

    # Save to cache
    print(f"Saving processed data to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], le.classes_)

    return X_train, y_train, X_test, test_ids, le
