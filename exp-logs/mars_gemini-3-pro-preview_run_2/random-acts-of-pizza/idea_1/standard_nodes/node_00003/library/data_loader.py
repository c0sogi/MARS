import os
import json
import pandas as pd
from library.config import (
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    DEBUG_MODE,
    DEBUG_SAMPLE_SIZE,
)


def load_raw_json(path):
    """
    Loads a JSON file into a pandas DataFrame.

    Args:
        path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the JSON data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def load_metadata(path):
    """
    Loads a metadata CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def load_and_merge_data(debug=DEBUG_MODE, debug_size=DEBUG_SAMPLE_SIZE):
    """
    Loads raw JSON data and merges it with metadata to create train, validation, and test sets.

    This function reads the raw JSON files (containing features) and the metadata CSV files
    (containing split definitions and labels). It merges them on 'request_id' to construct
    the final datasets.

    Args:
        debug (bool): If True, returns a subset of the data for debugging purposes.
        debug_size (int): The number of samples to return for each split in debug mode.

    Returns:
        tuple: A tuple containing three pandas DataFrames: (train_df, val_df, test_df).
    """
    # 1. Load Raw Data
    # We load the full raw datasets. 'train.json' contains data for both training and validation splits.
    df_raw_train = load_raw_json(TRAIN_JSON_PATH)
    df_raw_test = load_raw_json(TEST_JSON_PATH)

    # 2. Load Metadata
    # Metadata files define the specific request_ids for each split and contain the ground truth labels.
    meta_train = load_metadata(TRAIN_META_PATH)
    meta_val = load_metadata(VAL_META_PATH)
    meta_test = load_metadata(TEST_META_PATH)

    # 3. Pre-process Raw Data for Merging
    # The metadata for train/val contains the target label 'requester_received_pizza'.
    # The raw train data also contains this column. To avoid column duplication (e.g., _x, _y suffixes)
    # and ensure we use the verified labels from the metadata, we drop the target from the raw DataFrame.
    if "requester_received_pizza" in df_raw_train.columns:
        df_raw_train = df_raw_train.drop(columns=["requester_received_pizza"])

    # 4. Merge Data
    # We perform a left join on the metadata. This ensures that:
    #   a) We only keep rows that belong to the specific split (train vs val).
    #   b) The order of rows in metadata is preserved.
    #   c) We attach the raw features to the split definitions.

    train_df = pd.merge(meta_train, df_raw_train, on="request_id", how="left")
    val_df = pd.merge(meta_val, df_raw_train, on="request_id", how="left")

    # For the test set, the metadata does not contain the label, and neither does the raw test file.
    test_df = pd.merge(meta_test, df_raw_test, on="request_id", how="left")

    # 5. Handle Debug Mode
    if debug:
        train_df = train_df.head(debug_size)
        val_df = val_df.head(debug_size)
        test_df = test_df.head(debug_size)

    return train_df, val_df, test_df
