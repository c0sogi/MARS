import os
import numpy as np
import pandas as pd
from library.config import METADATA_DIR, IDEA_DIR, FEATURE_COLS, TARGET_COL


def load_datasets(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.
    Implements caching to ./working/idea_27/ to avoid re-parsing CSVs.
    Enforces strict float64 precision for feature matrices.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        dict: A dictionary containing:
            - 'X_train': Training features (np.float64)
            - 'y_train': Training labels (encoded int)
            - 'X_val': Validation features (np.float64)
            - 'y_val': Validation labels (encoded int)
            - 'X_test': Test features (np.float64)
            - 'ids_test': Test image IDs
            - 'classes': Array of class names (strings) corresponding to label indices
    """
    # Ensure the cache directory exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(IDEA_DIR, "X_train.npy"),
        "y_train": os.path.join(IDEA_DIR, "y_train.npy"),
        "X_val": os.path.join(IDEA_DIR, "X_val.npy"),
        "y_val": os.path.join(IDEA_DIR, "y_val.npy"),
        "X_test": os.path.join(IDEA_DIR, "X_test.npy"),
        "ids_test": os.path.join(IDEA_DIR, "ids_test.npy"),
        "classes": os.path.join(IDEA_DIR, "classes.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading datasets from cache at {IDEA_DIR}...")
            try:
                data = {}
                data["X_train"] = np.load(cache_files["X_train"])
                data["y_train"] = np.load(cache_files["y_train"])
                data["X_val"] = np.load(cache_files["X_val"])
                data["y_val"] = np.load(cache_files["y_val"])
                data["X_test"] = np.load(cache_files["X_test"])
                data["ids_test"] = np.load(cache_files["ids_test"])
                data["classes"] = np.load(cache_files["classes"], allow_pickle=True)
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing data...")
        else:
            print("Cache not found. Processing data from scratch...")
    else:
        print("Ignoring cache. Processing data from scratch...")

    # Load Metadata CSVs
    print("Reading metadata CSV files...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Extract Features (Strict float64)
    print("Extracting features...")
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)
    X_val = df_val[FEATURE_COLS].values.astype(np.float64)
    X_test = df_test[FEATURE_COLS].values.astype(np.float64)

    # Process Targets
    print("Encoding targets...")
    # We derive classes from the training set. Since it's a stratified split,
    # it should contain all classes. We sort them to ensure deterministic index mapping.
    classes = np.unique(df_train[TARGET_COL].values)
    classes.sort()

    # Create mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Encode labels
    y_train = np.array([class_to_idx[c] for c in df_train[TARGET_COL]], dtype=int)
    y_val = np.array([class_to_idx[c] for c in df_val[TARGET_COL]], dtype=int)

    # Extract Test IDs
    ids_test = df_test["id"].values

    # Save to Cache
    print(f"Saving processed datasets to {IDEA_DIR}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["classes"], classes)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "ids_test": ids_test,
        "classes": classes,
    }
