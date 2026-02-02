import os
import pandas as pd
from joblib import Parallel, delayed
from library.config import Config
from library.feature_engineering import extract_features_for_segment


def load_metadata(path):
    """
    Loads the metadata CSV file.

    Args:
        path (str): Path to the metadata file.

    Returns:
        pd.DataFrame: Loaded metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def _process_single_file(row, is_test):
    """
    Helper function to process a single file.
    Designed to be used with joblib for parallel processing.

    Args:
        row (pd.Series): A row from the metadata DataFrame.
        is_test (bool): Whether this is test data (no target).

    Returns:
        dict: Extracted features or None if failed.
    """
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

    try:
        if not os.path.exists(file_path):
            # Return None to be filtered out later
            return None

        # Load sensor data
        # Using float32 to handle potential nulls and save memory
        df = pd.read_csv(file_path, dtype="float32")

        # Extract features using the provided library function
        features = extract_features_for_segment(df)

        # Add identifiers
        features["segment_id"] = int(row["segment_id"])

        # Add target if available and not test set
        if not is_test and "time_to_eruption" in row:
            features["time_to_eruption"] = row["time_to_eruption"]

        return features

    except Exception:
        # Silently fail for individual files to prevent crashing the whole job
        return None


def process_dataset(
    meta_path, load_cached_data=True, is_test=False, debug_size=None, n_jobs=-1
):
    """
    Loads metadata, processes all segments to extract features in parallel, and handles caching.

    Args:
        meta_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether processing test data (no target column).
        debug_size (int, optional): Limit number of files processed for debugging.
        n_jobs (int): Number of parallel jobs. -1 uses all available cores.

    Returns:
        pd.DataFrame: Processed features dataframe.
    """
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache filename based on metadata filename and debug flag
    meta_name = os.path.basename(meta_path).replace(".csv", "")
    debug_suffix = f"_debug_{debug_size}" if debug_size else ""
    cache_filename = f"{meta_name}_features{debug_suffix}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Process from Scratch
    print(f"Processing data from {meta_path} (Debug Size: {debug_size})...")

    meta_df = load_metadata(meta_path)

    if debug_size is not None:
        meta_df = meta_df.head(debug_size)

    # Parallel execution
    # We iterate over rows. iterrows returns (index, series)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_file)(row, is_test) for _, row in meta_df.iterrows()
    )

    # Filter out None results (failed files)
    valid_results = [res for res in results if res is not None]

    if not valid_results:
        print("Warning: No features extracted.")
        return pd.DataFrame()

    # Create DataFrame
    features_df = pd.DataFrame(valid_results)

    # 3. Save to Cache
    print(f"Saving features to {cache_path}...")
    features_df.to_parquet(cache_path, index=False)

    return features_df
