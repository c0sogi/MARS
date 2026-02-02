import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.utils import set_seed

# Set global seed
set_seed(42)


def get_alphanumeric_features(columns):
    """
    Identifies feature columns and sorts them using standard lexicographical string sorting.
    This enforces Alphanumeric memory layout (e.g., 'margin_10' precedes 'margin_2').

    Args:
        columns (list): List of column names from the dataframe.

    Returns:
        list: Sorted list of feature column names.
    """
    # Filter columns that start with the feature prefixes
    feature_cols = [c for c in columns if c.startswith(("margin", "shape", "texture"))]

    # Standard lexicographical sort enforces the required alphanumeric ordering
    # Example: ['margin_1', 'margin_10', 'margin_11', ..., 'margin_2']
    return sorted(feature_cols)


def load_data(cache_dir="./working/idea_33/", load_cached_data=True):
    """
    Loads, preprocesses, and caches the dataset.
    Implements Alphanumeric Feature Ordering and Inductive Preprocessing.

    Args:
        cache_dir (str): Directory to save/load cached files.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.parquet"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.parquet"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.parquet"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # 1. Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = pd.read_parquet(cache_files["X_train"]).values
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_val = pd.read_parquet(cache_files["X_val"]).values
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            X_test = pd.read_parquet(cache_files["X_test"]).values
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Identify Feature Columns with Alphanumeric Ordering
    feature_cols = get_alphanumeric_features(train_df.columns)
    print(f"Features selected: {len(feature_cols)} (Alphanumeric Ordering)")

    # Extract Data (High Precision float64)
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df["species"].values

    X_val = val_df[feature_cols].values.astype(np.float64)
    y_val = val_df["species"].values

    X_test = test_df[feature_cols].values.astype(np.float64)
    test_ids = test_df["id"].values

    # Inductive Preprocessing Pipeline
    # Fit only on Train, then transform Train, Val, and Test

    # Step A: Yeo-Johnson Power Transformation
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train = pt.fit_transform(X_train)
    X_val = pt.transform(X_val)
    X_test = pt.transform(X_test)

    # Step B: Standard Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    # Get Classes
    classes = np.unique(y_train)

    # Save to Cache
    print("Saving processed data to cache...")
    pd.DataFrame(X_train, columns=feature_cols).to_parquet(cache_files["X_train"])
    np.save(cache_files["y_train"], y_train)

    pd.DataFrame(X_val, columns=feature_cols).to_parquet(cache_files["X_val"])
    np.save(cache_files["y_val"], y_val)

    pd.DataFrame(X_test, columns=feature_cols).to_parquet(cache_files["X_test"])
    np.save(cache_files["test_ids"], test_ids)

    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
