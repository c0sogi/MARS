import os
import pandas as pd
import numpy as np
from library.config import CACHE_DIR
from library.data_loader import DataLoader
from library.descriptors import process_single_structure


def generate_features(
    split="train", load_cached_data=True, debug=False, sample_size=50
):
    """
    Generates features for a given dataset split by iterating over metadata,
    computing descriptors for each structure, and merging them with the metadata.
    Implements caching to parquet files to avoid redundant computation.

    Args:
        split (str): The dataset split to process ('train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load features from the cache.
        debug (bool): If True, processes only a small subsample of the data.
        sample_size (int): The number of samples to process if debug is True.

    Returns:
        pd.DataFrame: A DataFrame containing the original metadata and the computed features.
    """
    # Determine cache file path
    suffix = "_debug" if debug else ""
    cache_file = os.path.join(CACHE_DIR, f"{split}_features{suffix}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute features from scratch
    print(f"Generating features for {split} set (debug={debug})...")

    # Initialize DataLoader and load metadata
    data_loader = DataLoader()
    metadata_df = data_loader.load_metadata(
        split=split, debug=debug, sample_size=sample_size
    )

    feature_list = []

    # Iterate over each material in the metadata
    for idx, row in metadata_df.iterrows():
        # Compute descriptors for the single structure
        # process_single_structure handles geometry loading and descriptor calculation
        feats = process_single_structure(row, data_loader)

        if feats is None:
            # In case of error (e.g., missing file), append empty dict (results in NaNs)
            feature_list.append({})
        else:
            feature_list.append(feats)

    # Convert list of feature dictionaries to DataFrame
    features_df = pd.DataFrame(feature_list)

    # Ensure index alignment with metadata for correct merging
    features_df.index = metadata_df.index

    # 3. Merge computed features with original metadata (composition, lattice vectors, targets)
    combined_df = pd.concat([metadata_df, features_df], axis=1)

    # 4. Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving features to {cache_file}")
    combined_df.to_parquet(cache_file, index=False)

    return combined_df


def clean_features(train_df, val_df=None, test_df=None):
    """
    Removes constant columns from the datasets.
    The columns to drop are determined based on the training set to prevent data leakage.

    Args:
        train_df (pd.DataFrame): Training features.
        val_df (pd.DataFrame, optional): Validation features.
        test_df (pd.DataFrame, optional): Test features.

    Returns:
        tuple: (cleaned_train_df, cleaned_val_df, cleaned_test_df)
    """
    print("Cleaning features: Dropping constant columns...")

    # Identify columns with more than 1 unique value in the training set
    # We assume 'id' and targets are not constant, but this check is safe for them too.
    keep_cols = train_df.columns[train_df.nunique() > 1].tolist()

    cleaned_train = train_df[keep_cols]

    cleaned_val = None
    if val_df is not None:
        # Keep only the columns that were kept in training
        # Intersection handles cases where val might miss a column (unlikely)
        common_cols = [c for c in keep_cols if c in val_df.columns]
        cleaned_val = val_df[common_cols]

    cleaned_test = None
    if test_df is not None:
        common_cols = [c for c in keep_cols if c in test_df.columns]
        cleaned_test = test_df[common_cols]

    n_dropped = len(train_df.columns) - len(cleaned_train.columns)
    print(f"Dropped {n_dropped} constant columns.")

    return cleaned_train, cleaned_val, cleaned_test
