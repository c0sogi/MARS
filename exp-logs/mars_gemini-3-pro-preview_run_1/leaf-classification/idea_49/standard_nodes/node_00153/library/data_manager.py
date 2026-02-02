import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from library.config import Config
from library.feature_extraction import extract_inertial_features

# Set random seed for reproducibility
np.random.seed(Config.SEED)


def load_and_merge_data(
    metadata_path, cache_name, load_cached_data=True, sample_size=None
):
    """
    Loads metadata, orchestrates parallel feature extraction, merges with tabular data,
    and handles caching.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_name (str): Base name for the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): Number of samples to process for debugging.

    Returns:
        pd.DataFrame: The merged and processed dataframe.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Modify cache name if sampling is active to avoid overwriting full cache
    if sample_size is not None:
        cache_name = f"{cache_name}_debug_{sample_size}"

    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # Fallback to recompute if cache load fails
            pass

    # 2. Compute from Scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Apply sampling if requested
    if sample_size is not None:
        meta_df = meta_df.head(sample_size)

    # Prepare image paths
    # Metadata contains relative path 'images/123.jpg', construct full path
    paths = [
        os.path.join(Config.INPUT_DIR, row["file_path"])
        for _, row in meta_df.iterrows()
    ]

    # Orchestrate Parallel Feature Extraction
    # Use joblib to parallelize the imported extract_inertial_features function
    extracted_dicts = Parallel(n_jobs=-1)(
        delayed(extract_inertial_features)(p) for p in paths
    )

    extracted_df = pd.DataFrame(extracted_dicts)

    # Merge Logic
    # Identify columns to keep from metadata (ID, Species)
    id_col = "id"
    target_col = "species"

    meta_cols_to_keep = [id_col]
    if target_col in meta_df.columns:
        meta_cols_to_keep.append(target_col)

    # Identify raw tabular features present in metadata
    raw_cols = [c for c in Config.RAW_TABULAR_FEATURES if c in meta_df.columns]

    # Concatenate metadata subset and extracted features
    # Reset index to ensure alignment (Parallel preserves order)
    df_final = pd.concat(
        [
            meta_df[meta_cols_to_keep + raw_cols].reset_index(drop=True),
            extracted_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Enforce Deterministic Alphanumeric Column Ordering
    # Separate ID/Target from feature columns for sorting
    feature_cols = raw_cols + list(extracted_df.columns)
    feature_cols = sorted(list(set(feature_cols)))  # Sort alphanumerically

    final_cols = meta_cols_to_keep + feature_cols
    df_final = df_final[final_cols]

    # 3. Save to Cache
    df_final.to_parquet(cache_path, index=False)

    return df_final


def get_train_data(load_cached_data=True, sample_size=None):
    """
    Wrapper to get training data using the parallel loader.
    """
    return load_and_merge_data(
        Config.TRAIN_METADATA_PATH,
        "train_data",
        load_cached_data=load_cached_data,
        sample_size=sample_size,
    )


def get_val_data(load_cached_data=True, sample_size=None):
    """
    Wrapper to get validation data using the parallel loader.
    """
    return load_and_merge_data(
        Config.VAL_METADATA_PATH,
        "val_data",
        load_cached_data=load_cached_data,
        sample_size=sample_size,
    )


def get_test_data(load_cached_data=True, sample_size=None):
    """
    Wrapper to get test data using the parallel loader.
    """
    return load_and_merge_data(
        Config.TEST_METADATA_PATH,
        "test_data",
        load_cached_data=load_cached_data,
        sample_size=sample_size,
    )
