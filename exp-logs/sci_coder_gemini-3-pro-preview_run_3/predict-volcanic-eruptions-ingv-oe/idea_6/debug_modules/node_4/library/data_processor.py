import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library import config
from library.features import extract_features_for_segment
from library.utils import load_metadata


def _process_wrapper(row):
    """
    Internal wrapper to call feature extraction on a single row.
    Used for parallelization.

    Args:
        row (pd.Series): A row from the metadata DataFrame.

    Returns:
        dict: Extracted features.
    """
    segment_id = int(row["segment_id"])
    # Metadata file_path is relative to INPUT_DIR (e.g., "train/123.csv")
    full_path = os.path.join(config.INPUT_DIR, row["file_path"])
    return extract_features_for_segment(full_path, segment_id)


def generate_dataset(metadata_df, dataset_name, load_cached_data=True, n_jobs=12):
    """
    Generates a feature dataset from metadata using parallel processing.
    Handles caching via Parquet files.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'segment_id' and 'file_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to load from parquet cache if available.
        n_jobs (int): Number of parallel jobs to use.

    Returns:
        pd.DataFrame: Feature matrix including 'segment_id'.
    """
    # Append debug suffix to cache file to avoid polluting full cache with sample data
    suffix = "_debug" if config.DEBUG else ""
    cache_filename = f"{dataset_name}_features{suffix}.parquet"
    cache_path = os.path.join(config.WORKING_DIR, cache_filename)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Compute from Scratch
    print(
        f"Generating features for {dataset_name} set ({len(metadata_df)} files) using {n_jobs} workers..."
    )

    # Prepare rows for parallel execution
    rows = [row for _, row in metadata_df.iterrows()]

    # Execute parallel feature extraction
    # verbose=0 suppresses joblib output, as requested
    feature_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_wrapper)(row) for row in rows
    )

    # Create DataFrame
    features_df = pd.DataFrame(feature_list)

    # Ensure segment_id is integer for clean merging later
    if "segment_id" in features_df.columns:
        features_df["segment_id"] = features_df["segment_id"].astype(int)

    # 3. Save Cache
    print(f"Saving features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    features_df.to_parquet(cache_path, index=False)

    return features_df


def get_train_val_datasets(load_cached_data=True):
    """
    Orchestrates the loading of training and validation datasets.

    Args:
        load_cached_data (bool): Whether to use cached features.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    train_meta, val_meta, _ = load_metadata()

    # Apply Debug Sampling
    if config.DEBUG:
        print(
            f"DEBUG: Sampling {config.DEBUG_SAMPLE_SIZE} rows for training and validation."
        )
        train_meta = train_meta.sample(
            n=min(len(train_meta), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        )
        val_meta = val_meta.sample(
            n=min(len(val_meta), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        )

    # Generate Features
    train_features = generate_dataset(
        train_meta, "train", load_cached_data=load_cached_data
    )
    val_features = generate_dataset(val_meta, "val", load_cached_data=load_cached_data)

    # Merge with Targets
    # Metadata contains 'time_to_eruption', features contain 'segment_id'
    train_merged = pd.merge(
        train_features,
        train_meta[["segment_id", "time_to_eruption"]],
        on="segment_id",
        how="left",
    )
    val_merged = pd.merge(
        val_features,
        val_meta[["segment_id", "time_to_eruption"]],
        on="segment_id",
        how="left",
    )

    # Separate Features (X) and Target (y)
    drop_cols = ["segment_id", "time_to_eruption"]

    X_train = train_merged.drop(columns=drop_cols)
    y_train = train_merged["time_to_eruption"]

    X_val = val_merged.drop(columns=drop_cols)
    y_val = val_merged["time_to_eruption"]

    return X_train, y_train, X_val, y_val


def get_test_dataset(load_cached_data=True):
    """
    Orchestrates the loading of the test dataset.

    Args:
        load_cached_data (bool): Whether to use cached features.

    Returns:
        tuple: (X_test, test_ids)
    """
    _, _, test_meta = load_metadata()

    # Apply Debug Sampling
    if config.DEBUG:
        print(f"DEBUG: Sampling {config.DEBUG_SAMPLE_SIZE} rows for testing.")
        test_meta = test_meta.sample(
            n=min(len(test_meta), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        )

    # Generate Features
    test_features = generate_dataset(
        test_meta, "test", load_cached_data=load_cached_data
    )

    # Separate Features (X) and IDs
    test_ids = test_features["segment_id"]
    X_test = test_features.drop(columns=["segment_id"])

    return X_test, test_ids
