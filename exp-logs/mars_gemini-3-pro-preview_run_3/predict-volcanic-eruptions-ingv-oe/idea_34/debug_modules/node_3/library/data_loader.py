import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library.config import Config
from library.feature_extraction import process_segment


def load_metadata(path):
    """
    Loads the metadata CSV file.

    Args:
        path (str): Path to the metadata CSV.

    Returns:
        pd.DataFrame: Loaded metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    return pd.read_csv(path)


def generate_dataset(
    metadata_path,
    cfg,
    load_cached_data=True,
    dataset_name="train",
    n_jobs=-1,
    sample_size=None,
):
    """
    Generates the dataset by processing sensor segments in parallel.
    Implements caching using Parquet files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cfg (Config): Configuration object.
        load_cached_data (bool): Whether to load from cache if available.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        n_jobs (int): Number of parallel jobs. -1 uses all available cores.
        sample_size (int, optional): Number of samples to process (for debugging).

    Returns:
        tuple: (X, y, segment_ids)
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or None): Target variable.
            segment_ids (pd.Series): Segment IDs.
    """
    # Ensure working directory exists
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.WORKING_DIR, f"{dataset_name}_features.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {dataset_name} data from {cache_path}...")
        df = pd.read_parquet(cache_path)

    # 2. Process from Scratch
    else:
        print(f"Processing {dataset_name} data from scratch...")
        meta_df = load_metadata(metadata_path)

        if sample_size is not None:
            print(f"Subsampling {sample_size} files for debugging...")
            meta_df = meta_df.head(sample_size)

        # Prepare parallel execution
        # We iterate over the metadata rows to get file paths and segment IDs
        tasks = []
        for _, row in meta_df.iterrows():
            full_path = os.path.join(cfg.INPUT_DIR, row["file_path"])
            tasks.append((full_path, row["segment_id"]))

        # Execute parallel processing
        print(f"Starting parallel feature extraction with n_jobs={n_jobs}...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_segment)(fp, sid, cfg) for fp, sid in tasks
        )

        # Filter out any failed processings (None results)
        results = [r for r in results if r is not None]

        if not results:
            raise ValueError("No features were extracted. Check input data and paths.")

        # Create DataFrame from list of dicts
        df = pd.DataFrame(results)

        # Merge target variable if it exists in metadata
        # process_segment returns features and segment_id, but not time_to_eruption
        if "time_to_eruption" in meta_df.columns:
            # Create a mapping from segment_id to target
            target_map = dict(zip(meta_df["segment_id"], meta_df["time_to_eruption"]))
            df["time_to_eruption"] = df["segment_id"].map(target_map)

        # Save to cache
        print(f"Saving {dataset_name} data to {cache_path}...")
        df.to_parquet(cache_path, index=False)

    # 3. Prepare Return Values
    segment_ids = df["segment_id"]

    if "time_to_eruption" in df.columns:
        y = df["time_to_eruption"]
        X = df.drop(columns=["segment_id", "time_to_eruption"])
    else:
        y = None
        X = df.drop(columns=["segment_id"])

    return X, y, segment_ids
