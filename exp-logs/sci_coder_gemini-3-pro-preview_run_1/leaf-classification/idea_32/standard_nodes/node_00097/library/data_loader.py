import os
import pandas as pd
import numpy as np
from library import config


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Implements caching using Parquet files in the working directory.
    Enforces strict feature ordering and float64 precision.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False or cache miss, re-processes raw metadata.

    Returns:
        tuple: ((X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids))
            - X_*: pandas DataFrame containing the 192 features (float64).
            - y_*: pandas Series containing the target species labels (string). None for test.
            - ids_*: pandas Series containing the image IDs (int).
    """

    # Define cache file paths
    cache_train_path = os.path.join(config.WORKING_DIR, "train_cache.parquet")
    cache_val_path = os.path.join(config.WORKING_DIR, "val_cache.parquet")
    cache_test_path = os.path.join(config.WORKING_DIR, "test_cache.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        try:
            df_train = pd.read_parquet(cache_train_path)
            df_val = pd.read_parquet(cache_val_path)
            df_test = pd.read_parquet(cache_test_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data.")
            return _process_and_cache_data(
                cache_train_path, cache_val_path, cache_test_path
            )
    else:
        print("Processing datasets from metadata...")
        df_train, df_val, df_test = _process_and_cache_data(
            cache_train_path, cache_val_path, cache_test_path
        )

    # Extract components
    # Train
    X_train = df_train[config.FEATURES]
    y_train = df_train["species"]
    train_ids = df_train["id"]

    # Val
    X_val = df_val[config.FEATURES]
    y_val = df_val["species"]
    val_ids = df_val["id"]

    # Test
    X_test = df_test[config.FEATURES]
    # Test set does not have 'species' column in the provided metadata/files usually,
    # or it might be ignored. The metadata generator creates test.csv without targets
    # if they aren't in the input. If they are, we ignore them for X.
    # We return ids for submission alignment.
    test_ids = df_test["id"]

    return (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids)


def _process_and_cache_data(train_out_path, val_out_path, test_out_path):
    """
    Internal function to load raw metadata, process features, and save to cache.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Debugging Subsampling
    if config.DEBUG:
        print(f"DEBUG MODE: Subsampling to {config.DEBUG_SAMPLE_SIZE} rows.")

        # Filter to top classes to ensure density (Cite debug_lesson_1)
        top_classes = df_train["species"].value_counts().index[:60]
        df_train = df_train[df_train["species"].isin(top_classes)]
        df_val = df_val[df_val["species"].isin(top_classes)]

        # Select validation set first
        df_val = df_val.head(config.DEBUG_SAMPLE_SIZE).copy()
        val_species = df_val["species"].unique()

        # Construct training set to guarantee coverage of validation species (Cite debug_lesson_6)
        df_train_subset = df_train[df_train["species"].isin(val_species)]
        train_guaranteed = df_train_subset.groupby("species").head(1)

        remaining = config.DEBUG_SAMPLE_SIZE - len(train_guaranteed)
        if remaining > 0:
            train_rest = df_train_subset.drop(train_guaranteed.index)
            train_fill = train_rest.head(remaining)
            df_train = pd.concat([train_guaranteed, train_fill])
        else:
            df_train = train_guaranteed.head(config.DEBUG_SAMPLE_SIZE)

        # Shuffle to remove ordering artifacts
        df_train = df_train.sample(frac=1, random_state=config.SEED).reset_index(
            drop=True
        )
        df_test = df_test.head(config.DEBUG_SAMPLE_SIZE).copy()

    # 3. Enforce Feature Consistency and Precision
    # We explicitly select columns based on the sorted FEATURES list in config.
    # This handles both ordering and filtering of extraneous columns.

    def process_df(df, is_test=False):
        # Verify all required features exist
        missing_features = [f for f in config.FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features[:5]}...")

        # Cast features to float64 explicitly
        for col in config.FEATURES:
            df[col] = df[col].astype(np.float64)

        # Keep only relevant columns for the cache file to save space/confusion
        cols_to_keep = ["id"] + config.FEATURES
        if not is_test:
            cols_to_keep.append("species")

        return df[cols_to_keep]

    df_train_proc = process_df(df_train, is_test=False)
    df_val_proc = process_df(df_val, is_test=False)
    df_test_proc = process_df(df_test, is_test=True)

    # 4. Save to Cache
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    df_train_proc.to_parquet(train_out_path, index=False)
    df_val_proc.to_parquet(val_out_path, index=False)
    df_test_proc.to_parquet(test_out_path, index=False)

    print(f"Datasets cached to {config.WORKING_DIR}")

    return df_train_proc, df_val_proc, df_test_proc
