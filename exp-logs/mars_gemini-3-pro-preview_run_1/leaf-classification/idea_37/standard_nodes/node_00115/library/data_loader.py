import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library import config, image_features


def load_and_augment_data(load_cached_data=True):
    """
    Loads tabular data, extracts and fuses geometric image features, and prepares
    high-precision float64 datasets for the Augmented OAS Discriminant pipeline.

    Implements caching to ./working/idea_37/ to ensure efficiency and reproducibility.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
            - X_*: pandas.DataFrame in float64 with sorted columns.
            - y_*: numpy.ndarray (int) encoded labels.
            - test_ids: numpy.ndarray (int) image IDs for submission.
            - classes: numpy.ndarray (str) original class names.
    """
    # Define cache file paths
    cache_paths = {
        "X_train": config.get_cache_path("X_train.parquet"),
        "y_train": config.get_cache_path("y_train.npy"),
        "X_val": config.get_cache_path("X_val.parquet"),
        "y_val": config.get_cache_path("y_val.npy"),
        "X_test": config.get_cache_path("X_test.parquet"),
        "test_ids": config.get_cache_path("test_ids.npy"),
        "classes": config.get_cache_path("classes.npy"),
    }

    # 1. Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_paths.values())
        if all_exist:
            print(f"Loading fused data from cache directory: {config.CACHE_DIR}")
            try:
                X_train = pd.read_parquet(cache_paths["X_train"])
                y_train = np.load(cache_paths["y_train"], allow_pickle=True)
                X_val = pd.read_parquet(cache_paths["X_val"])
                y_val = np.load(cache_paths["y_val"], allow_pickle=True)
                X_test = pd.read_parquet(cache_paths["X_test"])
                test_ids = np.load(cache_paths["test_ids"], allow_pickle=True)
                classes = np.load(cache_paths["classes"], allow_pickle=True)
                return X_train, y_train, X_val, y_val, X_test, test_ids, classes
            except Exception as e:
                print(f"Cache load failed ({e}). Recomputing from scratch...")
        else:
            print("Cache incomplete or missing. Computing from scratch...")

    # 2. Load Metadata
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)

    # 3. Extract Geometric Features (Augmentation)
    # This calls the library function which handles its own internal caching for the extraction step
    print("Extracting/Loading geometric features...")
    geo_train = image_features.extract_geometry(df_train, "train", load_cached_data)
    geo_val = image_features.extract_geometry(df_val, "val", load_cached_data)
    geo_test = image_features.extract_geometry(df_test, "test", load_cached_data)

    # 4. Fuse Data (Merge Tabular + Visual)
    print("Fusing tabular and visual features...")
    # Merge on 'id'. 'how=left' preserves the metadata order, though we will sort columns later.
    df_train_fused = pd.merge(df_train, geo_train, on="id", how="left")
    df_val_fused = pd.merge(df_val, geo_val, on="id", how="left")
    df_test_fused = pd.merge(df_test, geo_test, on="id", how="left")

    # 5. Process Targets (Label Encoding)
    print("Encoding targets...")
    le = LabelEncoder()
    y_train = le.fit_transform(df_train_fused["species"])
    y_val = le.transform(df_val_fused["species"])
    classes = le.classes_

    # 6. Process Features
    print("Processing feature matrices (Sorting & Casting)...")

    # Define columns to drop to isolate features
    # Note: 'file_path' is in metadata, 'species' is target, 'id' is identifier
    cols_to_exclude = {"id", "species", "file_path"}

    # Identify feature columns
    # We use the columns from train as the reference
    feature_cols = [c for c in df_train_fused.columns if c not in cols_to_exclude]

    # CRITICAL: Sort columns alphanumerically.
    # This ensures deterministic memory layout for linear algebra operations.
    feature_cols.sort()

    print(f"Total features selected: {len(feature_cols)}")

    # Extract and cast to high precision float64
    X_train = df_train_fused[feature_cols].astype(config.FLOAT_PRECISION)
    X_val = df_val_fused[feature_cols].astype(config.FLOAT_PRECISION)
    X_test = df_test_fused[feature_cols].astype(config.FLOAT_PRECISION)

    test_ids = df_test_fused["id"].values

    # 7. Save to Cache
    print(f"Saving processed datasets to {config.CACHE_DIR}...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    try:
        X_train.to_parquet(cache_paths["X_train"], index=False)
        np.save(cache_paths["y_train"], y_train)

        X_val.to_parquet(cache_paths["X_val"], index=False)
        np.save(cache_paths["y_val"], y_val)

        X_test.to_parquet(cache_paths["X_test"], index=False)
        np.save(cache_paths["test_ids"], test_ids)
        np.save(cache_paths["classes"], classes)
        print("Caching complete.")
    except Exception as e:
        print(f"Warning: Failed to save cache ({e})")

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
