import json
import os
import pandas as pd
from library.config import Config
from library.utils import set_seed


def load_and_merge_data(
    debug: bool = Config.DEBUG, debug_size: int = Config.DEBUG_SAMPLE_SIZE
):
    """
    Loads raw JSON data and metadata CSVs, merges them to create
    train, validation, and test DataFrames.

    Args:
        debug (bool): Whether to run in debug mode (sample data).
        debug_size (int): Number of samples to keep in debug mode.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    set_seed()

    # Define paths
    train_meta_path = Config.TRAIN_META_PATH
    val_meta_path = Config.VAL_META_PATH
    test_meta_path = Config.TEST_META_PATH

    train_json_path = Config.TRAIN_JSON_PATH
    test_json_path = Config.TEST_JSON_PATH

    # Load Metadata
    # Metadata contains the split definitions and the target labels (for train/val)
    df_meta_train = pd.read_csv(train_meta_path)
    df_meta_val = pd.read_csv(val_meta_path)
    df_meta_test = pd.read_csv(test_meta_path)

    # Load Raw JSON Data
    # Raw data contains the text and numerical features
    with open(train_json_path, "r") as f:
        raw_train_list = json.load(f)
    df_raw_train = pd.DataFrame(raw_train_list)

    with open(test_json_path, "r") as f:
        raw_test_list = json.load(f)
    df_raw_test = pd.DataFrame(raw_test_list)

    # Helper function to merge metadata with raw data
    def merge_data(meta_df, raw_df):
        # Identify columns to pull from raw_df
        # We exclude columns that are already in meta_df to avoid conflicts
        # Specifically 'requester_received_pizza' is present in both but we trust metadata

        # Get list of columns in raw_df that are NOT in meta_df
        cols_to_add = raw_df.columns.difference(meta_df.columns).tolist()

        # We must include the merge key 'request_id' if it was removed by difference()
        if "request_id" not in cols_to_add:
            cols_to_add.append("request_id")

        # Merge on request_id
        merged = pd.merge(meta_df, raw_df[cols_to_add], on="request_id", how="left")
        return merged

    # Create Splits
    df_train = merge_data(df_meta_train, df_raw_train)
    df_val = merge_data(df_meta_val, df_raw_train)
    df_test = merge_data(df_meta_test, df_raw_test)

    # Apply Debug Sampling if requested
    if debug:
        df_train = df_train.iloc[:debug_size].copy()
        df_val = df_val.iloc[:debug_size].copy()
        df_test = df_test.iloc[:debug_size].copy()

    return df_train, df_val, df_test
