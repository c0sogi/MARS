import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from library.data_handler import load_metadata, load_geometry
from library.physics_descriptors import process_single_structure

# Constants
CACHE_DIR = "./working/idea_37"


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the feature matrix by filling NaNs/Infs and dropping constant columns.

    Args:
        df (pd.DataFrame): The raw feature dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for training.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()

    # Fill NaNs with 0.0 (assuming NaN means absence of feature, e.g. no specific bond type)
    df_clean = df_clean.fillna(0.0)

    # Replace infinite values with 0.0
    df_clean = df_clean.replace([np.inf, -np.inf], 0.0)

    # Drop constant columns (std == 0)
    # We exclude 'id' from this check
    cols_to_check = [c for c in df_clean.columns if c != "id"]
    if len(cols_to_check) > 0:
        # Calculate standard deviation for numerical columns
        # Select only numeric types to avoid errors with object columns if any exist
        numeric_df = df_clean[cols_to_check].select_dtypes(include=[np.number])
        if not numeric_df.empty:
            std = numeric_df.std()
            constant_cols = std[std == 0].index.tolist()
            if constant_cols:
                df_clean = df_clean.drop(columns=constant_cols)

    return df_clean


def _process_wrapper(atoms):
    """
    Helper function for multiprocessing to safely call the descriptor calculator.
    """
    try:
        return process_single_structure(atoms)
    except Exception as e:
        # In production, logging this error would be good.
        # Returning None allows filtering out failed structures later.
        return None


def generate_feature_matrix(
    split: str, load_cached_data: bool = True, n_jobs: int = 4
) -> pd.DataFrame:
    """
    Generates the feature matrix for a specific dataset split.
    Uses multiprocessing to compute descriptors and caches the result.

    Args:
        split (str): Dataset split ('train', 'val', or 'test').
        load_cached_data (bool): If True, attempts to load from cache first.
        n_jobs (int): Number of parallel processes to use. If < 1, uses all available cores.

    Returns:
        pd.DataFrame: DataFrame containing features and the 'id' column.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{split}_features.parquet")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached features from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing features...")

    # Load metadata to get IDs and file paths
    df_meta = load_metadata(split)

    # Load ASE Atoms objects
    atoms_list = load_geometry(df_meta)

    # Determine number of processes
    if n_jobs < 1:
        n_jobs = cpu_count()

    print(
        f"Generating features for {len(atoms_list)} structures using {n_jobs} cores..."
    )

    # Parallel processing
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_process_wrapper, atoms_list)

    # Aggregate results
    valid_features = []

    for i, res in enumerate(results):
        if res is not None:
            # Attach ID to the feature dictionary
            res["id"] = df_meta.iloc[i]["id"]
            valid_features.append(res)
        else:
            # Handle failure by adding a row with just ID (will be filled with 0s by clean_features)
            # This ensures we don't lose rows, though they might be empty features
            valid_features.append({"id": df_meta.iloc[i]["id"]})

    # Create DataFrame
    df_features = pd.DataFrame(valid_features)

    # Ensure 'id' is integer
    if "id" in df_features.columns:
        df_features["id"] = df_features["id"].astype(int)

    # Save to cache
    try:
        df_features.to_parquet(cache_path, index=False)
        print(f"Features saved to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

    return df_features
