import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Constants
CACHE_DIR = "./working/idea_11"
METADATA_DIR = "./metadata"


def load_and_preprocess(load_cached_data=True):
    """
    Loads and preprocesses the leaf dataset.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
                                 If False or cache missing, reprocesses data.

    Returns:
        X_train (np.ndarray): Scaled feature matrix for training (combined train+val).
        y_train (np.ndarray): Encoded target labels for training.
        X_test (np.ndarray): Scaled feature matrix for testing.
        test_ids (np.ndarray): IDs for the test set.
        le (LabelEncoder): Fitted label encoder object.
    """

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
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
            print("Cache miss or incomplete. Reprocessing data...")
    else:
        print("Ignoring cache. Reprocessing data...")

    # 2. Load Metadata
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError("Metadata files not found in ./metadata/")

    df_train_part = pd.read_csv(train_path)
    df_val_part = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Combine train and val for maximum data utilization
    df_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # 3. Extract Features and Targets
    # Exclude non-feature columns
    exclude_cols = {"id", "species", "image_path"}
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Ensure consistent column ordering
    feature_cols.sort()

    X_train_raw = df_train[feature_cols].values
    y_train_raw = df_train["species"].values

    X_test_raw = df_test[feature_cols].values
    test_ids = df_test["id"].values

    # 4. Preprocessing

    # Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    # Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Save to Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], le.classes_)

    print(f"Data processed and saved to {CACHE_DIR}")

    return X_train, y_train, X_test, test_ids, le
