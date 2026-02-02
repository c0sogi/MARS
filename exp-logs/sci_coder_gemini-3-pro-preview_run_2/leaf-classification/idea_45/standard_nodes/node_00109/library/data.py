import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    FLOAT_PRECISION,
    CACHE_TRAIN_GLOBAL,
    CACHE_VAL_GLOBAL,
    CACHE_TEST_GLOBAL,
    CACHE_TRAIN_PHYSICAL,
    CACHE_VAL_PHYSICAL,
    CACHE_TEST_PHYSICAL,
    CACHE_Y_TRAIN,
    CACHE_Y_VAL,
    CACHE_CLASSES,
    CACHE_TEST_IDS,
    WORKING_DIR,
)
from library.features import generate_physical_dataset


def _get_feature_columns(df):
    """Identifies the 192 provided feature columns."""
    # Filter columns that start with margin, shape, or texture
    cols = [c for c in df.columns if c.startswith(("margin", "shape", "texture"))]
    # Sort to ensure consistent order (though usually they are ordered in CSV)
    # The provided CSVs have them, we just need to extract them.
    # We trust the CSV order but sorting by name ensures margin_1, margin_10, etc.
    # Actually, standard string sort might mess up _1 vs _10.
    # Let's rely on the list comprehension order which preserves CSV order usually.
    return cols


def _process_global_features(df, cache_path, load_cached_data=True):
    """
    Extracts, casts, and caches the provided 192 global features.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading global features from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            if data.shape[0] == len(df):
                return data.astype(FLOAT_PRECISION)
            else:
                print(
                    f"Cache shape mismatch ({data.shape[0]} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print(f"Processing global features for {len(df)} samples...")
    cols = _get_feature_columns(df)
    # Extract values and cast to float64
    data = df[cols].values.astype(FLOAT_PRECISION)

    print(f"Saving global features to cache: {cache_path}")
    np.save(cache_path, data)

    return data


def _process_targets(df_train, df_val, load_cached_data=True):
    """
    Encodes and caches targets and classes.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Check if all target caches exist
    if (
        load_cached_data
        and os.path.exists(CACHE_Y_TRAIN)
        and os.path.exists(CACHE_Y_VAL)
        and os.path.exists(CACHE_CLASSES)
    ):

        print("Loading targets and classes from cache...")
        try:
            y_train = np.load(CACHE_Y_TRAIN)
            y_val = np.load(CACHE_Y_VAL)
            classes = np.load(CACHE_CLASSES, allow_pickle=True)

            if len(y_train) == len(df_train) and len(y_val) == len(df_val):
                return y_train, y_val, classes
            else:
                print("Target cache shape mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading target cache: {e}. Recomputing...")

    print("Encoding targets...")
    le = LabelEncoder()
    # Fit on training species
    y_train = le.fit_transform(df_train["species"])
    classes = le.classes_

    # Transform validation species
    # Note: Stratified split ensures val classes are in train, but robust code handles potential issues
    y_val = le.transform(df_val["species"])

    print("Saving targets to cache...")
    np.save(CACHE_Y_TRAIN, y_train)
    np.save(CACHE_Y_VAL, y_val)
    np.save(CACHE_CLASSES, classes)

    return y_train, y_val, classes


def _process_test_ids(df_test, load_cached_data=True):
    """
    Extracts and caches test IDs.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(CACHE_TEST_IDS):
        try:
            ids = np.load(CACHE_TEST_IDS)
            if len(ids) == len(df_test):
                return ids
        except Exception:
            pass

    ids = df_test["id"].values
    np.save(CACHE_TEST_IDS, ids)
    return ids


def load_data(load_cached_data=True):
    """
    Main function to load all datasets (Global and Physical views).

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        dict: Dictionary containing:
            - X_train_global, X_val_global, X_test_global
            - X_train_physical, X_val_physical, X_test_physical
            - y_train, y_val
            - test_ids
            - classes
    """
    print("Loading Metadata...")
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # 1. Process Global Features (Provided 192 features)
    print("\n--- Processing Global View ---")
    X_train_global = _process_global_features(
        df_train, CACHE_TRAIN_GLOBAL, load_cached_data
    )
    X_val_global = _process_global_features(df_val, CACHE_VAL_GLOBAL, load_cached_data)
    X_test_global = _process_global_features(
        df_test, CACHE_TEST_GLOBAL, load_cached_data
    )

    # 2. Process Physical Features (Extracted Morphometrics)
    print("\n--- Processing Physical View ---")
    # generate_physical_dataset handles its own caching logic using the provided path
    X_train_physical = generate_physical_dataset(
        df_train, CACHE_TRAIN_PHYSICAL, load_cached_data
    )
    X_val_physical = generate_physical_dataset(
        df_val, CACHE_VAL_PHYSICAL, load_cached_data
    )
    X_test_physical = generate_physical_dataset(
        df_test, CACHE_TEST_PHYSICAL, load_cached_data
    )

    # 3. Process Targets
    print("\n--- Processing Targets ---")
    y_train, y_val, classes = _process_targets(df_train, df_val, load_cached_data)

    # 4. Process Test IDs
    test_ids = _process_test_ids(df_test, load_cached_data)

    return {
        "X_train_global": X_train_global,
        "X_val_global": X_val_global,
        "X_test_global": X_test_global,
        "X_train_physical": X_train_physical,
        "X_val_physical": X_val_physical,
        "X_test_physical": X_test_physical,
        "y_train": y_train,
        "y_val": y_val,
        "test_ids": test_ids,
        "classes": classes,
    }
