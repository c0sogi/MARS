import os
import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from library.config import (
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    NUMERICAL_FEATURES,
    TEXT_COLS,
)


def load_raw_json(path: str):
    """
    Loads raw JSON data from the specified path.
    """
    with open(path, "r") as f:
        return json.load(f)


def _process_split(meta_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges metadata with raw data, selects relevant columns, and handles text NaNs.

    Args:
        meta_df (pd.DataFrame): Metadata DataFrame containing request_ids and labels.
        raw_df (pd.DataFrame): Raw data DataFrame containing features.

    Returns:
        pd.DataFrame: Merged and filtered DataFrame.
    """
    # Define columns to extract from raw data (excluding label, which comes from metadata)
    # We always need the join key 'request_id'
    cols_to_extract = ["request_id"] + NUMERICAL_FEATURES + TEXT_COLS

    # Filter raw_df to only available columns to prevent KeyErrors
    available_cols = [c for c in cols_to_extract if c in raw_df.columns]
    raw_subset = raw_df[available_cols]

    # Merge metadata with raw subset on request_id
    # Metadata is the source of truth for the split and labels
    merged_df = meta_df.merge(raw_subset, on="request_id", how="left")

    # Fill missing values in text columns with empty strings
    for col in TEXT_COLS:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("").astype(str)
        else:
            merged_df[col] = ""

    return merged_df


def load_and_preprocess_data(load_cached_data: bool = True):
    """
    Loads, merges, and preprocesses the dataset (Train, Validation, and Test).

    Implements caching to avoid re-processing raw JSONs.
    Performs median imputation on numerical features (fit on train, transform all).

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_train = os.path.join(WORKING_DIR, "train_cleaned.parquet")
    cache_val = os.path.join(WORKING_DIR, "val_cleaned.parquet")
    cache_test = os.path.join(WORKING_DIR, "test_cleaned.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            print("Loading cached cleaned data...")
            return (
                pd.read_parquet(cache_train),
                pd.read_parquet(cache_val),
                pd.read_parquet(cache_test),
            )

    print("Processing data from scratch...")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Load Raw Data
    # train.json contains the full training set (which we split into train/val via metadata)
    raw_train_data = load_raw_json(TRAIN_JSON_PATH)
    df_raw_train = pd.DataFrame(raw_train_data)

    # test.json contains the test set
    raw_test_data = load_raw_json(TEST_JSON_PATH)
    df_raw_test = pd.DataFrame(raw_test_data)

    # 3. Load Metadata
    meta_train = pd.read_csv(TRAIN_META_PATH)
    meta_val = pd.read_csv(VAL_META_PATH)
    meta_test = pd.read_csv(TEST_META_PATH)

    # 4. Merge and Align
    df_train = _process_split(meta_train, df_raw_train)
    df_val = _process_split(meta_val, df_raw_train)
    df_test = _process_split(meta_test, df_raw_test)

    # 5. Impute Missing Numerical Values
    # We fit the imputer ONLY on the training set to avoid data leakage.
    imputer = SimpleImputer(strategy="median")

    # Fit on train
    df_train[NUMERICAL_FEATURES] = imputer.fit_transform(df_train[NUMERICAL_FEATURES])

    # Transform validation and test
    df_val[NUMERICAL_FEATURES] = imputer.transform(df_val[NUMERICAL_FEATURES])
    df_test[NUMERICAL_FEATURES] = imputer.transform(df_test[NUMERICAL_FEATURES])

    # 6. Ensure correct data types for labels
    if "requester_received_pizza" in df_train.columns:
        df_train["requester_received_pizza"] = df_train[
            "requester_received_pizza"
        ].astype(int)
    if "requester_received_pizza" in df_val.columns:
        df_val["requester_received_pizza"] = df_val["requester_received_pizza"].astype(
            int
        )

    # 7. Cache the results
    df_train.to_parquet(cache_train, index=False)
    df_val.to_parquet(cache_val, index=False)
    df_test.to_parquet(cache_test, index=False)

    return df_train, df_val, df_test
