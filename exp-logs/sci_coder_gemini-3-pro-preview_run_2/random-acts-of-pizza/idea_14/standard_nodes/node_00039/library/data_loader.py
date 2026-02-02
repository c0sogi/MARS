import json
import os
import pandas as pd
from library.config import (
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
)
from library.utils import validate_data_integrity


def load_raw_json(path):
    """
    Loads raw data from a JSON file into a pandas DataFrame.

    Args:
        path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the raw data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def load_split_metadata(split):
    """
    Loads metadata for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: DataFrame containing the metadata (ids and labels).
    """
    if split == "train":
        path = TRAIN_META_PATH
    elif split == "val":
        path = VAL_META_PATH
    elif split == "test":
        path = TEST_META_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def get_dataset(split, debug=False, debug_size=100):
    """
    Loads and aligns the dataset for a specific split by merging metadata with raw data.

    Args:
        split (str): The dataset split to load ('train', 'val', 'test').
        debug (bool): If True, loads only a subset of the data for debugging.
        debug_size (int): Number of samples to load in debug mode.

    Returns:
        pd.DataFrame: The aligned dataset containing features and labels (if available).
    """
    # 1. Load Metadata
    df_meta = load_split_metadata(split)

    if debug:
        df_meta = df_meta.head(debug_size)
        print(f"[{split.upper()}] Debug mode: Loaded {len(df_meta)} metadata records.")

    # 2. Determine Raw Source File
    # Train and Validation splits come from train.json
    # Test split comes from test.json
    if split in ["train", "val"]:
        raw_path = TRAIN_JSON_PATH
    elif split == "test":
        raw_path = TEST_JSON_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    # 3. Load Raw Data
    df_raw = load_raw_json(raw_path)

    # 4. Merge Metadata with Raw Data
    # We use a left merge on the metadata to ensure we only get the samples
    # defined in our split. Suffixes handle potential column name collisions.
    df_merged = df_meta.merge(
        df_raw, on="request_id", how="left", suffixes=("", "_raw")
    )

    # 5. Clean up
    # If the target column exists in both (e.g. from raw train.json),
    # we prefer the one from metadata (which is verified) and drop the raw duplicate.
    if "requester_received_pizza_raw" in df_merged.columns:
        df_merged.drop(columns=["requester_received_pizza_raw"], inplace=True)

    # Ensure the target is properly typed for training splits
    if split in ["train", "val"] and "requester_received_pizza" in df_merged.columns:
        df_merged["requester_received_pizza"] = df_merged[
            "requester_received_pizza"
        ].astype(int)

    # 6. Validate Integrity
    validate_data_integrity(
        df_merged, name=f"{split}_dataset", expected_rows=len(df_meta)
    )

    return df_merged
