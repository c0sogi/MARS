import os
import pandas as pd
from joblib import Parallel, delayed
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
)
from library.feature_engineering import process_segment_file
from library.utils import save_parquet, load_parquet


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = TRAIN_META_PATH
    elif split == "val":
        path = VAL_META_PATH
    elif split == "test":
        path = TEST_META_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    return pd.read_csv(path)


def create_dataset(
    split: str, load_cached_data: bool = True, debug_size: int = None, n_jobs: int = -1
) -> pd.DataFrame:
    """
    Generates or loads the dataset for a specific split.

    This function handles:
    1. Caching: Checks for existing parquet files to avoid re-computation.
    2. Metadata Loading: Reads the correct CSV based on the split.
    3. Parallel Processing: Uses joblib to extract features from sensor files in parallel.
    4. Assembly: Combines extracted features with segment_ids and targets.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.
        debug_size (int, optional): If provided, only processes the first N rows.
        n_jobs (int): Number of parallel jobs for feature extraction. -1 uses all cores.

    Returns:
        pd.DataFrame: The processed dataset containing features and targets.
    """
    # Determine cache filename based on split and debug status
    # We use different cache files for debug runs to prevent overwriting the full dataset cache
    if debug_size is not None:
        cache_filename = f"{split}_features_debug_{debug_size}.parquet"
    else:
        cache_filename = f"{split}_features.parquet"

    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from cache: {cache_path}")
        return load_parquet(cache_path)

    # 2. Compute from scratch
    print(f"Processing {split} dataset (debug_size={debug_size})...")

    # Load metadata
    meta_df = load_metadata(split)

    # Apply debug slicing if requested
    if debug_size is not None:
        meta_df = meta_df.head(debug_size)

    # Prepare full file paths for the feature extractor
    # The metadata contains relative paths (e.g., "train/1000015382.csv")
    paths = [os.path.join(INPUT_DIR, row["file_path"]) for _, row in meta_df.iterrows()]

    # Execute parallel feature extraction
    # We use verbose=0 to suppress progress bars as required
    feature_dicts = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_segment_file)(p) for p in paths
    )

    # Assemble the final DataFrame
    valid_records = []

    # Iterate through metadata and results together to align IDs and Targets
    for (idx, row), feats in zip(meta_df.iterrows(), feature_dicts):
        if feats is not None:
            # Start with the extracted features
            record = feats.copy()

            # Add the segment_id
            record["segment_id"] = row["segment_id"]

            # Add the target if it exists (train/val sets)
            if "time_to_eruption" in row:
                record["time_to_eruption"] = row["time_to_eruption"]

            valid_records.append(record)

    df = pd.DataFrame(valid_records)

    # 3. Save to cache for future runs
    print(f"Saving {split} features to cache: {cache_path}")
    save_parquet(df, cache_path)

    return df
