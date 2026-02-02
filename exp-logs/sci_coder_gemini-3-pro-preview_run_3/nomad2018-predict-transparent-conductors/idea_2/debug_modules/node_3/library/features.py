import os
import numpy as np
import pandas as pd
import ase.io
from library.config import Config

# Ensure reproducible results
np.random.seed(Config.RANDOM_SEED)


def compute_physical_descriptors(atoms):
    """
    Computes physical descriptors for a given ASE atoms object.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: A dictionary containing 'volume', 'density', and 'num_atoms'.
    """
    try:
        # Volume in Angstrom^3
        vol = atoms.get_volume()

        # Number of atoms
        n_atoms = len(atoms)

        # Density (Atomic Mass Units / Angstrom^3)
        # 1 AMU/A^3 approx 1.66 g/cm^3. We use the raw value as a feature.
        mass = sum(atoms.get_masses())
        density = mass / vol if vol > 1e-6 else 0.0

        return {"volume": vol, "density": density, "num_atoms": n_atoms}
    except Exception as e:
        print(f"Error computing descriptors: {e}")
        return {"volume": 0.0, "density": 0.0, "num_atoms": 0}


class GNNFeatureExtractor:
    """
    Placeholder for GNN Feature Extractor.
    Disabled due to broken DGL installation in the environment.
    """

    def __init__(self, model_name=None, device=None):
        print("GNNFeatureExtractor is disabled due to DGL environment issues.")

    def extract_features(self, atoms_list, batch_size=None):
        """
        Returns empty features.
        """
        print("Skipping GNN feature extraction (DGL unavailable).")
        # Return an empty array with shape (n_samples, 0)
        return np.empty((len(atoms_list), 0))


def process_data(metadata_path, cache_path, load_cached_data=True):
    """
    Main data processing function. Loads metadata, extracts features (physical + GNN),
    and returns a combined DataFrame. Implements caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path to save/load the Parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed feature matrix.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Debugging: Sample subset if configured
    if Config.DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()

    # 3. Extract Features
    # Lists to store results
    physical_feats = []
    atoms_objects = []
    valid_indices = []

    print("Reading geometry files and computing physical descriptors...")
    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            try:
                atoms = ase.io.read(full_path)

                # Physical descriptors
                phys = compute_physical_descriptors(atoms)
                physical_feats.append(phys)

                # Store for GNN
                atoms_objects.append(atoms)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
        else:
            print(f"File not found: {full_path}")

    # Filter dataframe to valid rows
    df_valid = df.loc[valid_indices].reset_index(drop=True)

    # Create DataFrame from physical features
    df_phys = pd.DataFrame(physical_feats)

    # 4. GNN Feature Extraction - SKIPPED
    print("GNN features skipped. Using only tabular and physical features.")

    # 5. Combine All Features
    # Concatenate: [Metadata (Tabular) + Physical]
    # Drop file_path as it's not a feature
    df_final = pd.concat([df_valid.drop(columns=["file_path"]), df_phys], axis=1)

    # 6. Save to Cache
    print(f"Saving {len(df_final)} processed rows to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_final.to_parquet(cache_path, index=False)

    return df_final
