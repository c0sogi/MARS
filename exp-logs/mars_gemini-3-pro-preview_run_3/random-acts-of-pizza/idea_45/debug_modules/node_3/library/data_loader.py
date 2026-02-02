import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import load_data as utils_load_data


def load_dataset(split, load_cached_data=True, limit=None):
    """
    Loads and preprocesses the dataset for a specific split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load processed data from cache.
        limit (int, optional): If provided, limits the number of samples (for debugging).

    Returns:
        dict: A dictionary containing:
            - 'ids': np.array of request_ids
            - 'y': np.array of target labels (None for test)
            - 'metadata': pd.DataFrame of numerical metadata features
            - 'text': pd.Series of concatenated title and body text
            - 'community': pd.Series of subreddit lists
    """
    # Define cache file paths
    cache_prefix = os.path.join(Config.CACHE_DIR, f"{split}")
    path_metadata = f"{cache_prefix}_metadata.parquet"
    path_text = f"{cache_prefix}_text.parquet"
    path_community = f"{cache_prefix}_community.parquet"
    path_ids = f"{cache_prefix}_ids.npy"
    path_y = f"{cache_prefix}_labels.npy"

    # Check if all required cache files exist
    cache_exists = (
        os.path.exists(path_metadata)
        and os.path.exists(path_text)
        and os.path.exists(path_community)
        and os.path.exists(path_ids)
    )

    # For train/val, we also need the labels cache
    if split in ["train", "val"]:
        cache_exists = cache_exists and os.path.exists(path_y)

    # 1. Try to load from cache
    if load_cached_data and cache_exists:
        print(f"Loading {split} data from cache...")
        metadata = pd.read_parquet(path_metadata)
        # Load text and extract the series (saved as DataFrame for Parquet compatibility)
        text = pd.read_parquet(path_text)[Config.TEXT_COL]
        # Load community list
        community = pd.read_parquet(path_community)[Config.SUBREDDIT_LIST_COL]
        ids = np.load(path_ids, allow_pickle=True)

        y = None
        if split in ["train", "val"]:
            y = np.load(path_y)

    else:
        # 2. Process from scratch
        print(f"Processing {split} data from scratch...")

        # Load raw metadata using the provided utility
        df = utils_load_data(split)

        # Extract IDs
        ids = df[Config.ID_COL].values

        # Extract Target (if applicable)
        y = None
        if split in ["train", "val"]:
            y = df[Config.TARGET_COL].values

        # Text Processing: Concatenate Title and Body
        # Fill NaNs with empty strings to ensure string concatenation works
        title = df[Config.TITLE_COL].fillna("").astype(str)
        body = df[Config.TEXT_COL].fillna("").astype(str)
        text = title + " " + body

        # Community Processing: Extract subreddit lists
        community = df[Config.SUBREDDIT_LIST_COL]

        # Metadata Processing: Positive Feature Selection
        # Filter for allow-listed columns only
        valid_cols = [c for c in Config.METADATA_COLS if c in df.columns]
        metadata = df[valid_cols].copy()

        # Enforce float32 for numerical metadata to save memory
        for col in metadata.columns:
            metadata[col] = metadata[col].astype(np.float32)

        # 3. Save to cache
        print(f"Saving {split} data to cache at {Config.CACHE_DIR}...")

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        metadata.to_parquet(path_metadata, index=False)
        # Save Series as DataFrame to use Parquet
        pd.DataFrame({Config.TEXT_COL: text}).to_parquet(path_text, index=False)
        pd.DataFrame({Config.SUBREDDIT_LIST_COL: community}).to_parquet(
            path_community, index=False
        )
        np.save(path_ids, ids)

        if y is not None:
            np.save(path_y, y)

    # 4. Apply Limit (Debugging)
    if limit is not None:
        print(f"Limiting dataset to {limit} samples.")
        ids = ids[:limit]
        metadata = metadata.iloc[:limit]
        text = text.iloc[:limit]
        community = community.iloc[:limit]
        if y is not None:
            y = y[:limit]

    # Construct result dictionary
    result = {
        "ids": ids,
        "metadata": metadata,
        "text": text,
        "community": community,
        "y": y,
    }

    return result


def get_labels(split):
    """
    Retrieves the target labels for a given split.

    Args:
        split (str): 'train' or 'val'.

    Returns:
        np.array: The target labels.
    """
    if split == "test":
        raise ValueError("Test set does not contain labels.")

    # Load dataset (will use cache if available) and extract 'y'
    data = load_dataset(split, load_cached_data=True)
    return data["y"]
