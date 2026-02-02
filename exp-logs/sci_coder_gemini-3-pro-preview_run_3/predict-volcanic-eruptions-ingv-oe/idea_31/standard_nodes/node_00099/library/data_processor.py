import os
import pandas as pd
from joblib import Parallel, delayed
import library.config as config
import library.features as features


def process_set(metadata_path, cache_name, load_cached_data=True, sample_size=None):
    """
    Orchestrates the data processing pipeline for a given dataset (train/val/test).

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_name (str): Filename for the parquet cache (e.g., 'train_features.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): Number of segments to process for debugging purposes.

    Returns:
        pd.DataFrame: The processed feature matrix.
    """
    # Ensure the working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(config.WORKING_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        df = pd.read_parquet(cache_path)

        # If debugging with a smaller sample size, return just the head of the cached data
        if sample_size is not None:
            return df.iloc[:sample_size]
        return df

    # 2. Compute features if cache is missing or forced reload
    print(f"Processing data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Apply sampling if requested
    if sample_size is not None:
        print(f"Sampling {sample_size} segments for debugging...")
        meta_df = meta_df.iloc[:sample_size]

    # Execute feature extraction in parallel
    # Using 12 jobs to maximize utilization of available vCPUs
    results = Parallel(n_jobs=12, verbose=0)(
        delayed(features._process_file_wrapper)(row) for _, row in meta_df.iterrows()
    )

    # Filter out any failed processing attempts (None results)
    results = [r for r in results if r is not None]

    if not results:
        print("Warning: No data was successfully processed.")
        return pd.DataFrame()

    df_features = pd.DataFrame(results)

    # 3. Save to cache
    # Strictly save only if we processed the full dataset to maintain cache integrity
    if sample_size is None:
        print(f"Saving features to {cache_path}")
        df_features.to_parquet(cache_path, index=False)

    return df_features
