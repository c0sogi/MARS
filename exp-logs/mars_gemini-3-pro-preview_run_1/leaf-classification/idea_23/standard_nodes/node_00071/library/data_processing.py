import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config


def extract_genus(species_series):
    """
    Extracts the genus from a pandas Series of species names.
    Assumes format 'Genus_Species'.
    """
    return species_series.apply(lambda x: x.split("_")[0])


def get_pipeline():
    """
    Returns the preprocessing pipeline: Yeo-Johnson transformation followed by Standard Scaling.
    """
    # Note: standardize=False in PowerTransformer because we apply StandardScaler explicitly afterwards.
    return [PowerTransformer(method="yeo-johnson", standardize=False), StandardScaler()]


def process_data(load_cached_data=True):
    """
    Loads data, extracts taxonomy, performs inductive preprocessing, and caches results.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.

    Returns:
        tuple: (X_train, y_train, genus_train, X_val, y_val, genus_val, X_test, ids_test, classes)
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(config.WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(config.WORKING_DIR, "y_train.npy"),
        "genus_train": os.path.join(config.WORKING_DIR, "genus_train.npy"),
        "X_val": os.path.join(config.WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(config.WORKING_DIR, "y_val.npy"),
        "genus_val": os.path.join(config.WORKING_DIR, "genus_val.npy"),
        "X_test": os.path.join(config.WORKING_DIR, "X_test.npy"),
        "ids_test": os.path.join(config.WORKING_DIR, "ids_test.npy"),
        "classes": os.path.join(config.WORKING_DIR, "classes.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading cached data from {config.WORKING_DIR}...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            genus_train = np.load(cache_files["genus_train"], allow_pickle=True)
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            genus_val = np.load(cache_files["genus_val"], allow_pickle=True)
            X_test = np.load(cache_files["X_test"])
            ids_test = np.load(cache_files["ids_test"])
            classes = np.load(cache_files["classes"], allow_pickle=True)

            return (
                X_train,
                y_train,
                genus_train,
                X_val,
                y_val,
                genus_val,
                X_test,
                ids_test,
                classes,
            )

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load metadata CSVs
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_val = pd.read_csv(config.VAL_PATH)
    df_test = pd.read_csv(config.TEST_PATH)

    # Extract Targets and Taxonomy
    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values

    genus_train = extract_genus(df_train[config.TARGET_COL]).values
    genus_val = extract_genus(df_val[config.TARGET_COL]).values

    ids_test = df_test[config.ID_COL].values

    # Identify Classes (sorted for consistency)
    classes = np.sort(np.unique(y_train))

    # Extract Features (ensure deterministic order)
    X_train_raw = df_train[config.FEATURES].values.astype(config.DTYPE)
    X_val_raw = df_val[config.FEATURES].values.astype(config.DTYPE)
    X_test_raw = df_test[config.FEATURES].values.astype(config.DTYPE)

    # Inductive Preprocessing
    # Fit pipeline ONLY on training data
    pt, ss = get_pipeline()

    print("Fitting PowerTransformer on training data...")
    X_train_pt = pt.fit_transform(X_train_raw)

    print("Fitting StandardScaler on training data...")
    X_train_processed = ss.fit_transform(X_train_pt)

    # Transform validation and test data using fitted pipeline
    print("Transforming validation and test data...")
    X_val_processed = ss.transform(pt.transform(X_val_raw))
    X_test_processed = ss.transform(pt.transform(X_test_raw))

    # 3. Save to cache
    print(f"Saving processed data to {config.WORKING_DIR}...")
    np.save(cache_files["X_train"], X_train_processed)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["genus_train"], genus_train)

    np.save(cache_files["X_val"], X_val_processed)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["genus_val"], genus_val)

    np.save(cache_files["X_test"], X_test_processed)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["classes"], classes)

    return (
        X_train_processed,
        y_train,
        genus_train,
        X_val_processed,
        y_val,
        genus_val,
        X_test_processed,
        ids_test,
        classes,
    )
