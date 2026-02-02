import os
import warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library.config import WORKING_DIR, DEBUG_MODE, DEBUG_SAMPLE_SIZE
from library.data_loader import load_metadata, load_structure
from library.descriptors import StructureFeaturizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def _process_single_entry(id_val, file_path, featurizer):
    """
    Helper function to process a single structure.
    Executed in parallel workers.
    """
    try:
        atoms = load_structure(file_path)
        feats = featurizer.featurize(atoms)
        feats["id"] = id_val
        return feats
    except Exception as e:
        # Log error but don't crash the entire pipeline
        print(f"Error processing id {id_val}: {e}")
        return None


def generate_features(metadata_df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Generates features for the given metadata DataFrame using parallel processing.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
        n_jobs (int): Number of parallel jobs. -1 uses all available cores.

    Returns:
        pd.DataFrame: DataFrame with original metadata and new features merged on 'id'.
    """
    # Initialize the featurizer
    # It will be pickled and sent to workers
    featurizer = StructureFeaturizer()

    # Prepare tasks
    tasks = [
        (row["id"], row["file_path"], featurizer) for _, row in metadata_df.iterrows()
    ]

    # Execute in parallel using joblib
    # backend="loky" is robust for pickling complex objects
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_entry)(id_val, path, featurizer)
        for id_val, path, featurizer in tasks
    )

    # Filter out failed entries
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        raise RuntimeError("No features computed. Check input data or error logs.")

    # Create DataFrame from list of dictionaries
    feat_df = pd.DataFrame(valid_results)

    # Merge with original metadata to keep targets and other info
    # Inner join ensures we only keep rows where features were successfully computed
    merged_df = pd.merge(metadata_df, feat_df, on="id", how="inner")

    return merged_df


def process_split(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Main function to process a data split with caching.

    Args:
        split (str): The data split to process ('train', 'val', or 'test').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: Processed dataframe with features.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file path
    cache_file = os.path.join(WORKING_DIR, f"{split}_features_idea_40.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading cached features for {split} from {cache_file}")
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing features for {split} set (Parallel)...")

    # Load metadata
    meta_df = load_metadata(split, debug=DEBUG_MODE, sample_size=DEBUG_SAMPLE_SIZE)

    # Determine number of jobs (use fewer in debug mode to reduce overhead)
    n_jobs = 2 if DEBUG_MODE else -1

    # Generate features
    df_features = generate_features(meta_df, n_jobs=n_jobs)

    # 3. Save to cache
    print(f"Saving features to {cache_file}")
    try:
        df_features.to_parquet(cache_file, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache file: {e}")

    return df_features
