import os
import pandas as pd
import numpy as np
import library.config as config
from library.features import process_data
from library.utils import log1p_transform


def clean_features(df_train, df_val, df_test):
    """
    Identifies constant or quasi-constant columns in the training set
    and drops them from all datasets.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        df_test (pd.DataFrame): Test data.

    Returns:
        tuple: (df_train_clean, df_val_clean, df_test_clean)
    """
    # Identify feature columns (exclude targets and id)
    # We only look at columns present in train (which includes targets)
    # but we must exclude targets/id from the check.
    exclude = config.TARGET_COLS + ["id"]
    # Filter for numeric columns only to compute std
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Calculate std dev (pandas ignores NaNs by default)
    std = df_train[feature_cols].std()

    # Identify constant columns (std == 0)
    constant_cols = std[std == 0].index.tolist()

    # Identify all-NaN columns (count == 0)
    counts = df_train[feature_cols].count()
    empty_cols = counts[counts == 0].index.tolist()

    cols_to_drop = list(set(constant_cols + empty_cols))

    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} constant/empty columns...")
        # Drop from all datasets
        df_train = df_train.drop(columns=cols_to_drop)
        df_val = df_val.drop(columns=cols_to_drop)
        df_test = df_test.drop(columns=cols_to_drop)
    else:
        print("No constant columns found.")

    return df_train, df_val, df_test


def prepare_datasets(load_cached_data=True):
    """
    Loads features, cleans them, applies target transformations, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
                                 If False or load fails, re-processes from raw features.

    Returns:
        tuple: (train_df, val_df, test_df)
               DataFrames containing features, ids, and (for train/val) transformed targets.
    """
    # Define cache paths with debug suffix if applicable
    suffix = ""
    if config.DEBUG:
        suffix = f"_debug_{config.DEBUG_SAMPLE_SIZE}"

    cache_train = os.path.join(
        config.WORKING_DIR, f"train_prepared_idea35{suffix}.parquet"
    )
    cache_val = os.path.join(config.WORKING_DIR, f"val_prepared_idea35{suffix}.parquet")
    cache_test = os.path.join(
        config.WORKING_DIR, f"test_prepared_idea35{suffix}.parquet"
    )

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            print("Loading prepared datasets from cache...")
            try:
                train_df = pd.read_parquet(cache_train)
                val_df = pd.read_parquet(cache_val)
                test_df = pd.read_parquet(cache_test)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load prepared cache: {e}. Re-processing...")
        else:
            print("Prepared cache not found. Processing from features...")
    else:
        print("Force re-processing prepared datasets...")

    # 2. Load raw features
    # process_data handles its own caching of the raw feature extraction
    print("Loading raw features...")
    df_train_raw = process_data("train", load_cached_data=load_cached_data)
    df_val_raw = process_data("val", load_cached_data=load_cached_data)
    df_test_raw = process_data("test", load_cached_data=load_cached_data)

    # 3. Clean features
    print("Cleaning features...")
    df_train_clean, df_val_clean, df_test_clean = clean_features(
        df_train_raw, df_val_raw, df_test_raw
    )

    # 4. Transform targets (Log1p)
    print("Transforming targets...")
    for target in config.TARGET_COLS:
        if target in df_train_clean.columns:
            # Apply log1p transformation to stabilize variance
            df_train_clean[target] = log1p_transform(df_train_clean[target])
        if target in df_val_clean.columns:
            df_val_clean[target] = log1p_transform(df_val_clean[target])

    # 5. Save to cache
    print(f"Saving prepared datasets to {config.WORKING_DIR}...")
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    df_train_clean.to_parquet(cache_train, index=False)
    df_val_clean.to_parquet(cache_val, index=False)
    df_test_clean.to_parquet(cache_test, index=False)

    return df_train_clean, df_val_clean, df_test_clean
