import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library.config import load_data, extract_features, METADATA_DIR, WORKING_DIR
from library.utils import reduce_mem_usage


def load_metadata(mode: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The metadata DataFrame.
    """
    file_path = os.path.join(METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    return pd.read_csv(file_path)


def _process_single_item(segment_id, file_path, target=None):
    """
    Helper function to process a single data segment.
    Used for parallel execution.
    """
    try:
        # load_data expects relative path from input dir
        df_sensor = load_data(file_path)
        feats = extract_features(df_sensor)
        feats["segment_id"] = segment_id

        if target is not None:
            feats["time_to_eruption"] = target

        return feats
    except Exception as e:
        # Return None on failure to be filtered out later
        return None


def generate_dataset(
    mode: str, load_cached_data: bool = True, debug_size: int = None, n_jobs: int = 12
):
    """
    Generates the dataset for the specified mode, utilizing parallel processing.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet cache if available.
        debug_size (int, optional): Number of samples to process for debugging.
        n_jobs (int): Number of parallel jobs.

    Returns:
        tuple: (X, y)
            - If mode is 'train' or 'val': X is features DataFrame (clean), y is target Series.
            - If mode is 'test': X is features DataFrame (includes segment_id), y is None.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Construct cache filename
    cache_filename = f"{mode}_features.parquet"
    if debug_size:
        cache_filename = f"{mode}_features_debug_{debug_size}.parquet"

    cache_path = os.path.join(WORKING_DIR, cache_filename)

    df = None

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} features from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        # 2. Process from Scratch
        print(f"Processing {mode} data...")
        meta_df = load_metadata(mode)

        if debug_size:
            meta_df = meta_df.iloc[:debug_size]
            print(f"Debug mode: processing {len(meta_df)} samples.")

        # Prepare arguments for parallel execution
        tasks = []
        for _, row in meta_df.iterrows():
            seg_id = row["segment_id"]
            f_path = row["file_path"]
            target = row.get("time_to_eruption", None)
            tasks.append((seg_id, f_path, target))

        # Execute parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_item)(s, f, t) for s, f, t in tasks
        )

        # Filter out None results (errors)
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            raise RuntimeError("No data processed successfully.")

        df = pd.DataFrame(valid_results)

        # Optimize memory
        df = reduce_mem_usage(df, verbose=False)

        # 3. Save Cache
        print(f"Saving {mode} features to {cache_path}")
        df.to_parquet(cache_path, index=False)

    # Return X and y based on mode
    if mode == "test":
        # For test, we need segment_id for submission, so we keep it in X.
        # y is None.
        return df, None
    else:
        # For train/val, we separate features and target.
        # We drop segment_id as it's not a predictive feature.
        if "time_to_eruption" in df.columns:
            y = df["time_to_eruption"]
            X = df.drop(columns=["time_to_eruption", "segment_id"])
        else:
            y = None
            X = df.drop(columns=["segment_id"], errors="ignore")

        return X, y
