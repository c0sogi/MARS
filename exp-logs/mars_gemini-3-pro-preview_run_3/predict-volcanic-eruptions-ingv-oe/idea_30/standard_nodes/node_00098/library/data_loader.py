import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from library.config import Config
from library.features import extract_segment_features
from library.utils import load_sensor_data


def load_metadata(mode):
    """
    Loads the metadata CSV file for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: Metadata containing segment_ids and file_paths.
    """
    csv_path = os.path.join(Config.METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
    return pd.read_csv(csv_path)


def _process_single_row(row_dict):
    """
    Helper function to process a single row of metadata.
    Intended for use with parallel execution.

    Args:
        row_dict (dict): Dictionary containing metadata for one segment.

    Returns:
        dict: Extracted features or None if processing fails.
    """
    segment_id = row_dict["segment_id"]
    file_path = os.path.join(Config.INPUT_DIR, row_dict["file_path"])

    # Load sensor data
    df = load_sensor_data(file_path)

    # Skip if empty
    if df.empty:
        return None

    try:
        # Extract features using the library function
        feats = extract_segment_features(df, segment_id)

        # Add target if present in metadata
        if "time_to_eruption" in row_dict:
            feats["time_to_eruption"] = row_dict["time_to_eruption"]

        return feats
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def build_feature_dataset(mode="train", load_cached_data=True, debug_size=None):
    """
    Constructs the feature dataset for the given mode.
    Handles caching and parallel processing.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.
        debug_size (int, optional): Limit number of samples for debugging.

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a Series of targets (or None).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_features.parquet")

    df = None

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        try:
            df = pd.read_parquet(cache_file)
            if debug_size is not None:
                df = df.head(debug_size)
        except Exception as e:
            print(f"Failed to load cache: {e}. Proceeding to recompute.")
            df = None

    # 2. Compute from Scratch if needed
    if df is None:
        print(f"Processing {mode} data from scratch...")

        # Load Metadata
        meta_df = load_metadata(mode)

        if debug_size is not None:
            meta_df = meta_df.head(debug_size)

        # Prepare for parallel execution
        rows = meta_df.to_dict("records")
        # Use n_jobs from config, defaulting to 4 if not set
        n_workers = Config.LGBM_PARAMS.get("n_jobs", 4)

        results = []

        # Execute in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # map preserves order
            for res in executor.map(_process_single_row, rows):
                if res is not None:
                    results.append(res)

        df = pd.DataFrame(results)

        # Save to Cache (only if not debugging and dataframe is not empty)
        if debug_size is None and not df.empty:
            print(f"Saving {mode} features to {cache_file}...")
            df.to_parquet(cache_file, index=False)

    # 3. Structure Output (X, y)
    if df.empty:
        return pd.DataFrame(), None

    if "time_to_eruption" in df.columns:
        y = df["time_to_eruption"]
        # X contains segment_id and features, but not target
        X = df.drop(columns=["time_to_eruption"])
    else:
        y = None
        X = df

    return X, y
