import os
import json
import pandas as pd
from library.config import (
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
)


def load_dataset_with_metadata(split="train", load_cached_data=True):
    """
    Loads the dataset for the specified split (train, val, test), merging raw JSON
    data with the corresponding metadata CSV. Implements caching to parquet.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet cache if available.

    Returns:
        pd.DataFrame: The merged dataframe containing raw features and labels (if available).
    """
    # Ensure working directory exists (redundant with config but safe)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_path = os.path.join(WORKING_DIR, f"raw_data_{split}.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            # Using pyarrow engine to handle complex types (lists) if possible
            df = pd.read_parquet(cache_path, engine="pyarrow")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Determine paths based on split
    if split == "train":
        meta_path = TRAIN_META_PATH
        json_path = TRAIN_JSON_PATH
    elif split == "val":
        meta_path = VAL_META_PATH
        json_path = TRAIN_JSON_PATH
    elif split == "test":
        meta_path = TEST_META_PATH
        json_path = TEST_JSON_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # 3. Load Metadata
    print(f"Loading metadata from {meta_path}...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    df_meta = pd.read_csv(meta_path)

    # 4. Load Raw JSON
    print(f"Loading raw JSON from {json_path}...")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)
    df_json = pd.DataFrame(data)

    # 5. Merge Metadata with Raw Data
    # We perform a left merge on the metadata to select only the samples for this split
    # and to attach the correct labels/indices.
    print(f"Merging metadata and raw data for {split} split...")
    df = df_meta.merge(df_json, on="request_id", how="left")

    # 6. Clean up columns (Resolve conflicts between metadata and raw json)
    # Metadata usually contains the authoritative 'requester_received_pizza' (int)
    # JSON contains 'requester_received_pizza' (bool)
    # Merge creates _x (from meta) and _y (from json)

    if "requester_received_pizza_x" in df.columns:
        # Keep the metadata label (int) and rename it back
        df["requester_received_pizza"] = df["requester_received_pizza_x"]
        df.drop(columns=["requester_received_pizza_x"], inplace=True)

    if "requester_received_pizza_y" in df.columns:
        # Drop the raw json label to avoid confusion
        df.drop(columns=["requester_received_pizza_y"], inplace=True)

    # 7. Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    try:
        # Use pyarrow to handle potential list columns (e.g. subreddits list)
        df.to_parquet(cache_path, index=False, engine="pyarrow")
    except Exception as e:
        print(f"Warning: Failed to save cache for {split}: {e}")

    return df
