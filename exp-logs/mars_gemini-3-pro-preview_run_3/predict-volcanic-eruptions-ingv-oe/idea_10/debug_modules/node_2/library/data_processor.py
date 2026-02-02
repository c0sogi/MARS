import os
import pandas as pd
import joblib
from library.feature_engineering import process_segment
from library.config import METADATA_DIR, WORKING_DIR, N_JOBS


def generate_feature_matrix(dataset_type, load_cached=True, debug_size=None):
    """
    Generates or loads the feature matrix for a specific dataset type (train, val, test).

    Args:
        dataset_type (str): One of 'train', 'val', or 'test'.
        load_cached (bool): If True, attempts to load from cache first.
        debug_size (int, optional): If set, only processes the first N rows of metadata.
                                    Creates a separate cache file for debug runs.

    Returns:
        pd.DataFrame: The DataFrame containing extracted features and segment_ids.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Construct cache filename
    filename = f"{dataset_type}_features"
    if debug_size is not None:
        filename += f"_debug_{debug_size}"
    filename += ".parquet"

    cache_path = os.path.join(WORKING_DIR, filename)

    # 1. Try to load from cache
    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached features for {dataset_type} from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Compute from scratch
    print(f"Generating features for {dataset_type} (debug_size={debug_size})...")

    # Load metadata
    meta_path = os.path.join(METADATA_DIR, f"{dataset_type}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Apply debug sampling if requested
    if debug_size is not None:
        meta_df = meta_df.head(debug_size)
        print(f"Sampled metadata to {len(meta_df)} rows.")

    # Parallel feature extraction
    # We use the imported process_segment function which handles individual file loading and processing
    results = joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(process_segment)(row["file_path"], row["segment_id"])
        for _, row in meta_df.iterrows()
    )

    # Filter out any failed files (None results)
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        raise ValueError(
            f"No valid features were generated for {dataset_type}. Check input data or logs for errors."
        )

    features_df = pd.DataFrame(valid_results)

    # 3. Save to cache
    print(f"Saving features for {dataset_type} to {cache_path}...")
    features_df.to_parquet(cache_path)

    return features_df
