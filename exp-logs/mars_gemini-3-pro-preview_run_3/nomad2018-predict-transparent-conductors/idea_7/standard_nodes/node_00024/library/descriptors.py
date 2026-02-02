import os
import numpy as np
import pandas as pd
import ase.io
from library.config import Config


def compute_physical_properties(atoms):
    """
    Computes physical properties for a single ASE Atoms object.

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Dictionary containing 'volume', 'density', and 'avg_nn_dist'.
    """
    # 1. Volume (Angstrom^3)
    volume = atoms.get_volume()

    # 2. Density (Atomic Mass Units / Angstrom^3)
    # Note: This is proportional to g/cm^3.
    # We use sum of atomic masses divided by volume.
    total_mass = sum(atoms.get_masses())
    density = total_mass / volume if volume > 0 else 0.0

    # 3. Average Nearest Neighbor Distance
    # Calculate all-pairs distances with Minimum Image Convention (MIC) to account for PBC
    # get_all_distances(mic=True) returns an NxN matrix
    distances = atoms.get_all_distances(mic=True)

    # We want the nearest neighbor for each atom.
    # The distance matrix includes 0 on the diagonal (distance to self).
    # We replace 0 with infinity to ignore self-distance.
    np.fill_diagonal(distances, np.inf)

    # Find the minimum distance for each atom (distance to its nearest neighbor)
    min_distances = np.min(distances, axis=1)

    # Average these minimum distances to get a single scalar for the structure
    avg_nn_dist = np.mean(min_distances)

    # Calculate standard deviation of nearest neighbor distances to capture structural distortion
    std_nn_dist = np.std(min_distances)

    return {
        "volume": volume,
        "density": density,
        "avg_nn_dist": avg_nn_dist,
        "std_nn_dist": std_nn_dist,
    }


def extract_descriptors(metadata_df, split="train", load_cached_data=True):
    """
    Extracts explicit physical descriptors from raw geometry files.

    Features extracted:
    - volume: Unit cell volume.
    - density: Mass density proxy.
    - avg_nn_dist: Average nearest neighbor distance (bond length proxy).

    Implements caching using Parquet files in the working directory.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path' and 'id'.
        split (str): Dataset split name ('train', 'val', 'test') for naming cache files.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing the extracted features, indexed by original index.
    """
    # Construct cache file path
    cache_filename = f"{split}_physical_features.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached physical descriptors from {cache_path}")
        try:
            df_features = pd.read_parquet(cache_path)
            # Ensure the length matches. If metadata was subsampled for debugging,
            # the cache might be stale or larger/smaller.
            # For simplicity in this controlled environment, we assume cache validity
            # if it exists, or the user manages cache invalidation.
            # However, strictly checking length is safer.
            if len(df_features) == len(metadata_df):
                return df_features
            else:
                print(
                    f"Cache size mismatch ({len(df_features)} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting physical descriptors for {len(metadata_df)} samples...")

    features_list = []

    for idx, row in metadata_df.iterrows():
        # Construct full path to geometry file
        # row['file_path'] is relative, e.g., "train/1/geometry.xyz"
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # Read crystal structure
            atoms = ase.io.read(full_path, format="aims")

            # Compute properties
            props = compute_physical_properties(atoms)
            features_list.append(props)

        except Exception as e:
            # Handle potential read errors (though data is expected to be clean)
            print(f"Error processing {full_path}: {e}")
            # Append defaults or NaNs to maintain alignment
            features_list.append(
                {
                    "volume": np.nan,
                    "density": np.nan,
                    "avg_nn_dist": np.nan,
                    "std_nn_dist": np.nan,
                }
            )

    # Create DataFrame
    df_features = pd.DataFrame(features_list, index=metadata_df.index)

    # 3. Save to cache
    try:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df_features.to_parquet(cache_path, index=False)
        print(f"Saved physical descriptors to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return df_features
