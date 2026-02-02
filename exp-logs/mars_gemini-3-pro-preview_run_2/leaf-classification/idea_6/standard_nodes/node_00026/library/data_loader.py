import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, CACHE_DIR


def get_feature_columns(df):
    """
    Extracts feature column names corresponding to margin, shape, and texture.
    """
    return [c for c in df.columns if c.startswith(("margin", "shape", "texture"))]


def load_datasets(load_cached_data=True, combine_train_val=False):
    """
    Loads, preprocesses, and scales the dataset.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
        combine_train_val (bool): If True, merges training and validation sets for final training.
                                  If False, returns separate training and validation sets.

    Returns:
        tuple:
            If combine_train_val is False:
                (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
            If combine_train_val is True:
                (X_train_full, y_train_full, None, None, X_test, test_ids, classes)
    """
    # Define cache filenames based on mode
    suffix = "_combined" if combine_train_val else "_split"

    cache_files = {
        "X_train": os.path.join(CACHE_DIR, f"X_train{suffix}.npy"),
        "y_train": os.path.join(CACHE_DIR, f"y_train{suffix}.npy"),
        "X_test": os.path.join(CACHE_DIR, f"X_test{suffix}.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),  # Common
        "classes": os.path.join(CACHE_DIR, "classes.npy"),  # Common
    }

    if not combine_train_val:
        cache_files["X_val"] = os.path.join(CACHE_DIR, "X_val_split.npy")
        cache_files["y_val"] = os.path.join(CACHE_DIR, "y_val_split.npy")

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(
                f"Loading cached data from {CACHE_DIR} (Mode: {'Combined' if combine_train_val else 'Split'})..."
            )
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"], allow_pickle=True)

            if not combine_train_val:
                X_val = np.load(cache_files["X_val"])
                y_val = np.load(cache_files["y_val"], allow_pickle=True)
                return X_train, y_train, X_val, y_val, X_test, test_ids, classes
            else:
                return X_train, y_train, None, None, X_test, test_ids, classes
        else:
            print("Cached files not found or incomplete. Processing from scratch...")

    # 2. Process from scratch
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Extract IDs and Classes
    test_ids = df_test["id"].values

    # Get all unique classes sorted alphabetically
    # We combine train and val species to ensure we capture all (though they should be same due to stratification)
    classes = sorted(
        list(set(df_train["species"].unique()) | set(df_val["species"].unique()))
    )
    classes = np.array(classes)

    # Identify feature columns
    feature_cols = get_feature_columns(df_train)
    print(f"Identified {len(feature_cols)} feature columns.")

    # 3. Handle Splitting/Combining and Scaling
    scaler = StandardScaler()

    if combine_train_val:
        print("Combining Train and Validation sets for final training...")
        # Concatenate DataFrames
        df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

        X_train_raw = df_full[feature_cols].values
        y_train = df_full["species"].values
        X_test_raw = df_test[feature_cols].values

        # Fit scaler on full training data
        print("Fitting StandardScaler on combined dataset...")
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        X_val = None
        y_val = None

    else:
        print("Processing separate Train and Validation sets...")
        X_train_raw = df_train[feature_cols].values
        y_train = df_train["species"].values
        X_val_raw = df_val[feature_cols].values
        y_val = df_val["species"].values
        X_test_raw = df_test[feature_cols].values

        # Fit scaler ONLY on training data to prevent leakage
        print("Fitting StandardScaler on training split...")
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

    # 4. Save to cache
    print(f"Saving processed data to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    if not combine_train_val:
        np.save(cache_files["X_val"], X_val)
        np.save(cache_files["y_val"], y_val)

    print("Data processing complete.")

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
