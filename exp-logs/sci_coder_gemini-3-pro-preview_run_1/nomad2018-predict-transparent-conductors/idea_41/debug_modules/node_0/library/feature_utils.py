import os
import numpy as np
import pandas as pd
from library.config import Config


def calculate_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    Calculates the volume of the unit cell given lattice parameters.
    Vectorized for numpy arrays.

    Formula: V = abc * sqrt(1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))
    """
    alpha_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)
    gamma_rad = np.radians(gamma_deg)

    ca = np.cos(alpha_rad)
    cb = np.cos(beta_rad)
    cg = np.cos(gamma_rad)

    term = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    # Ensure non-negative before sqrt to handle potential floating point errors
    term = np.maximum(term, 0.0)

    volume = a * b * c * np.sqrt(term)
    return volume


def compute_global_features(df):
    """
    Extracts and computes global features from the metadata DataFrame.

    Features (12 dims):
    1-3. Lattice Vector Lengths (a, b, c)
    4-6. Lattice Angles (alpha, beta, gamma)
    7.   Unit Cell Volume
    8.   Atomic Density (Total Atoms / Volume)
    9-11. Stoichiometry (Al, Ga, In %)
    12.  Total Number of Atoms

    Args:
        df (pd.DataFrame): Metadata dataframe containing lattice and composition columns.

    Returns:
        np.ndarray: Array of shape (N, 12) containing global features.
    """
    # Extract raw columns
    a = df["lattice_vector_1_ang"].values
    b = df["lattice_vector_2_ang"].values
    c = df["lattice_vector_3_ang"].values

    alpha = df["lattice_angle_alpha_degree"].values
    beta = df["lattice_angle_beta_degree"].values
    gamma = df["lattice_angle_gamma_degree"].values

    n_atoms = df["number_of_total_atoms"].values

    pct_al = df["percent_atom_al"].values
    pct_ga = df["percent_atom_ga"].values
    pct_in = df["percent_atom_in"].values

    # Derived features
    volume = calculate_cell_volume(a, b, c, alpha, beta, gamma)

    # Avoid division by zero if volume is somehow 0 (unlikely in valid data)
    safe_volume = np.where(volume > 1e-6, volume, 1.0)
    density = n_atoms / safe_volume

    # Stack features in the order defined in Config
    # Shape: (N, 12)
    features = np.column_stack(
        [a, b, c, alpha, beta, gamma, volume, density, pct_al, pct_ga, pct_in, n_atoms]
    )

    return features.astype(np.float32)


def get_global_features(split, load_cached_data=True):
    """
    Retrieves global features for a specific data split ('train', 'val', 'test').
    Implements caching to avoid re-computation.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempt to load from cache first.

    Returns:
        np.ndarray: Global features array.
    """
    # Determine cache path
    cache_file = os.path.join(Config.WORKING_DIR, f"global_features_{split}.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached global features for '{split}' from {cache_file}")
        try:
            return np.load(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Determine metadata path
    if split == "train":
        meta_path = Config.TRAIN_METADATA
    elif split == "val":
        meta_path = Config.VAL_METADATA
    elif split == "test":
        meta_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Load metadata
    print(f"Computing global features for '{split}' from {meta_path}")
    df = pd.read_csv(meta_path)

    # Compute features
    features = compute_global_features(df)

    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, features)
    print(f"Saved global features to {cache_file}")

    return features
