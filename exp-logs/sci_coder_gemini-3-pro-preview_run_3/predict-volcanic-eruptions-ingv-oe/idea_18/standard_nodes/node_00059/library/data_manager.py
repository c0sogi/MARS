import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from library import config, feature_extraction


def load_metadata(dataset_type):
    """
    Loads the metadata CSV for the specified dataset type.

    Args:
        dataset_type (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: Metadata dataframe.
    """
    file_path = os.path.join(config.METADATA_DIR, f"{dataset_type}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found at {file_path}")
    return pd.read_csv(file_path)


def _process_file_wrapper(args):
    """
    Worker function for parallel processing.
    Loads a single CSV file and extracts features.

    Args:
        args (tuple): (segment_id, file_path, target)

    Returns:
        dict: Extracted features with segment_id and target (if available).
    """
    segment_id, rel_file_path, target = args
    full_path = os.path.join(config.INPUT_DIR, rel_file_path)

    try:
        # Load sensor data
        # Using float32 to match library configuration and optimize memory
        df = pd.read_csv(full_path, dtype="float32")

        # Extract features using the library function
        features = feature_extraction.extract_segment_features(df)

        # Append metadata
        features["segment_id"] = int(segment_id)

        # Append target if it exists (for train/val sets)
        if target is not None:
            features["time_to_eruption"] = target

        return features

    except Exception as e:
        print(f"Error processing {full_path}: {e}")
        return None


def generate_feature_matrix(dataset_type, debug=False, load_cached_data=True):
    """
    Generates the feature matrix for a given dataset type using multiprocessing.
    Implements caching to Parquet format.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        debug (bool): If True, processes only a small subset of data.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing features, segment_id, and (optionally) target.
    """
    # 1. Determine Cache Path
    cache_filename = f"{dataset_type}_features"
    if debug:
        cache_filename += "_debug"
    cache_path = os.path.join(config.WORKING_DIR, f"{cache_filename}.parquet")

    # 2. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {dataset_type} features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating {dataset_type} features (Debug={debug})...")

    # 3. Load Metadata
    meta_df = load_metadata(dataset_type)

    if debug:
        meta_df = meta_df.iloc[: config.DEBUG_SAMPLE_SIZE].copy()

    # 4. Prepare Tasks for Multiprocessing
    tasks = []
    for _, row in meta_df.iterrows():
        # Check if target column exists (it won't for test set)
        target = row["time_to_eruption"] if "time_to_eruption" in row else None
        tasks.append((row["segment_id"], row["file_path"], target))

    # 5. Execute Parallel Feature Extraction
    # Use all available CPUs (or a safe margin)
    n_processes = max(1, cpu_count())

    results = []
    with Pool(processes=n_processes) as pool:
        # imap_unordered is generally faster/more memory efficient for large iterables
        for res in pool.imap_unordered(_process_file_wrapper, tasks):
            if res is not None:
                results.append(res)

    if not results:
        raise RuntimeError(
            f"No features generated for {dataset_type}. Check input data."
        )

    # 6. Aggregate Results
    feature_df = pd.DataFrame(results)

    # 7. Save to Cache
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving features to {cache_path}...")
    feature_df.to_parquet(cache_path, index=False)

    return feature_df
