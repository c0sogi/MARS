import os
import numpy as np
import pandas as pd
import ase.io
from tqdm import tqdm
import warnings

# Import configuration and data loader
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_COMBINED_FEATURES_PATH,
    VAL_COMBINED_FEATURES_PATH,
    TEST_COMBINED_FEATURES_PATH,
    TABULAR_FEATURES,
    RANDOM_SEED,
)
from library.data_loader import load_metadata, load_geometry

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(RANDOM_SEED)


class PhysicalDescriptorExtractor:
    """
    Extracts explicit physical descriptors from the atomic geometry.
    Calculates volume, density, and validates atom counts.
    """

    def __init__(self):
        # Atomic masses in atomic mass units (u)
        self.atomic_masses = {"Al": 26.981539, "Ga": 69.723, "In": 114.818, "O": 15.999}

    def extract(self, atoms: ase.Atoms) -> dict:
        """
        Calculates physical properties from an ASE Atoms object.
        Cite solution_lesson_node_00018: Prioritize Raw Geometry Over Summary Metadata
        """
        if atoms is None:
            return {"volume": np.nan, "density": np.nan, "num_atoms_geometry": np.nan}

        # 1. Volume (Angstrom^3)
        try:
            volume = atoms.get_volume()
        except ValueError:
            # Fallback for non-periodic systems or errors
            volume = np.nan

        # 2. Mass and Density
        # Density units: u / A^3. To convert to g/cm^3, multiply by 1.66054
        total_mass = sum(atoms.get_masses())
        if volume > 0:
            density = total_mass / volume
        else:
            density = np.nan

        # 3. Number of atoms
        num_atoms = len(atoms)

        return {"volume": volume, "density": density, "num_atoms_geometry": num_atoms}


def generate_features(
    split: str, load_cached_data: bool = True, limit: int = None
) -> pd.DataFrame:
    """
    Main function to generate the feature matrix for a given data split.
    Combines metadata and physical descriptors.
    Handles caching to avoid re-computation.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        limit (int): Optional limit for debugging.

    Returns:
        pd.DataFrame: The complete feature matrix (X) combined with targets (y) if available.
    """
    # Determine cache path
    if split == "train":
        cache_path = TRAIN_COMBINED_FEATURES_PATH
    elif split == "val":
        cache_path = VAL_COMBINED_FEATURES_PATH
    elif split == "test":
        cache_path = TEST_COMBINED_FEATURES_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if limit:
            return df.head(limit)
        return df

    print(f"Generating features for {split} set (Cache miss or force reload)...")

    # 2. Load Metadata
    meta_df = load_metadata(split)
    if limit:
        meta_df = meta_df.head(limit)

    # 3. Initialize Extractors
    phys_extractor = PhysicalDescriptorExtractor()

    # 4. Iterate and Compute
    phys_features_list = []

    # Use tqdm for progress tracking
    print("Extracting physical features...")
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        # Load geometry
        atoms = load_geometry(row["file_path"])

        # Extract Physical Features
        phys_feats = phys_extractor.extract(atoms)
        phys_features_list.append(phys_feats)

    # 5. Create DataFrames
    phys_df = pd.DataFrame(phys_features_list, index=meta_df.index)

    # 6. Combine All Features
    # Start with tabular features from metadata
    # Ensure we only keep relevant tabular columns + targets + id
    cols_to_keep = ["id"] + TABULAR_FEATURES

    # Add targets if they exist (train/val)
    targets_exist = all(
        col in meta_df.columns
        for col in ["formation_energy_ev_natom", "bandgap_energy_ev"]
    )
    if targets_exist:
        cols_to_keep.extend(["formation_energy_ev_natom", "bandgap_energy_ev"])

    # Filter metadata
    base_df = meta_df[cols_to_keep].copy()

    # Concatenate horizontally
    # Cite solution_lesson_node_00016: Avoid Feature Dilution (Removed emb_df)
    final_df = pd.concat([base_df, phys_df], axis=1)

    # 7. Save to Cache
    print(f"Saving generated features to {cache_path}...")
    # Ensure directory exists (redundant but safe)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    final_df.to_parquet(cache_path, index=False)

    return final_df
