import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    TARGET_COLS,
)
from library.geometry_utils import extract_structural_features


def load_metadata(split="train", sample_size=None):
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        sample_size (int, optional): Number of rows to sample for debugging.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    if sample_size is not None and sample_size < len(df):
        # Use a fixed random state for reproducibility of the sample
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


def build_feature_matrix(metadata_df, split, load_cached_data=True):
    """
    Orchestrates the creation of the full feature matrix.
    It combines tabular metadata with geometric features extracted via geometry_utils.
    Implements caching for the final combined dataframe.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata (id, tabular features).
        split (str): The dataset split name (e.g., 'train', 'val', 'test') used for naming cache files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Combined dataframe with both tabular and geometric features.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache filenames
    # We use a specific name for the combined features to distinguish from the raw geometric features cache
    combined_cache_file = os.path.join(
        WORKING_DIR, f"{split}_combined_features.parquet"
    )

    # 1. Try to load the fully combined cache first
    if load_cached_data and os.path.exists(combined_cache_file):
        print(f"Loading combined features from {combined_cache_file}...")
        try:
            combined_df = pd.read_parquet(combined_cache_file)
            # Validate length
            if len(combined_df) == len(metadata_df):
                return combined_df
            else:
                print(
                    f"Combined cache length mismatch ({len(combined_df)} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load combined cache: {e}. Recomputing...")

    # 2. Compute/Load Geometric Features
    # We pass a specific cache name to extract_structural_features so it manages its own cache
    geo_cache_name = f"{split}_features.parquet"
    print(f"Extracting/Loading geometric features for {split} set...")
    geo_features = extract_structural_features(
        metadata_df, load_cached_data=load_cached_data, cache_file_name=geo_cache_name
    )

    # 3. Merge Geometric Features with Tabular Metadata
    print("Merging geometric features with tabular metadata...")

    # Ensure 'id' columns are integers for proper merging
    metadata_df["id"] = metadata_df["id"].astype(int)
    geo_features["id"] = geo_features["id"].astype(int)

    # Merge
    combined_df = pd.merge(metadata_df, geo_features, on="id", how="inner")

    # Drop columns that are not features or targets
    if "file_path" in combined_df.columns:
        combined_df = combined_df.drop(columns=["file_path"])

    # 4. Save Combined Cache
    print(f"Saving combined features to {combined_cache_file}...")
    try:
        combined_df.to_parquet(combined_cache_file)
    except Exception as e:
        print(f"Warning: Failed to save combined cache: {e}")

    return combined_df


def log_transform_targets(df, target_cols=TARGET_COLS):
    """
    Applies log1p transformation to the specified target columns in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing target columns.
        target_cols (list): List of column names to transform.

    Returns:
        pd.DataFrame: Dataframe with transformed targets.
    """
    df_transformed = df.copy()
    for col in target_cols:
        if col in df_transformed.columns:
            # log1p(x) = log(1 + x), suitable for non-negative energy values
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed


def inverse_transform_targets(pred_array):
    """
    Applies expm1 transformation to reverse the log1p transform.

    Args:
        pred_array (np.array): Array of log-transformed predictions.

    Returns:
        np.array: Array of predictions in original scale.
    """
    # expm1(x) = exp(x) - 1
    return np.expm1(pred_array)
