import os
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    RDF_BINS,
    RDF_MIN_DIST,
    RDF_MAX_DIST,
    ADF_BINS,
    ADF_MIN_ANGLE,
    ADF_MAX_ANGLE,
    CUTOFF_DISTANCE,
    RANDOM_SEED,
)

# Constants for atomic numbers
METALS = [13, 31, 49]  # Al, Ga, In
OXYGEN = 8
ALL_ELEMENTS = [8, 13, 31, 49]  # O, Al, Ga, In


def get_physical_properties(atoms: Atoms) -> dict:
    """
    Calculates physical properties of the structure: Volume, Density, Number of Atoms.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 1e-6 else 0.0
    return {"geo_volume": vol, "geo_density": density, "geo_num_atoms": len(atoms)}


def compute_rdf(atoms: Atoms, r_min: float, r_max: float, n_bins: int) -> dict:
    """
    Computes element-resolved Radial Distribution Functions (RDF).
    Returns a dictionary of histogram features.
    """
    # Define pairs of interest (sorted to handle symmetry)
    pairs = []
    for i in range(len(ALL_ELEMENTS)):
        for j in range(i, len(ALL_ELEMENTS)):
            pairs.append((ALL_ELEMENTS[i], ALL_ELEMENTS[j]))

    bins = np.linspace(r_min, r_max, n_bins + 1)
    histograms = {}

    # Get all pairwise distances with Minimum Image Convention
    # This returns a symmetric matrix of distances
    dist_matrix = atoms.get_all_distances(mic=True)
    atomic_numbers = atoms.get_atomic_numbers()

    for z1, z2 in pairs:
        idxs1 = np.where(atomic_numbers == z1)[0]
        idxs2 = np.where(atomic_numbers == z2)[0]

        if len(idxs1) == 0 or len(idxs2) == 0:
            hist = np.zeros(n_bins)
        else:
            # Extract relevant submatrix
            sub_mat = dist_matrix[np.ix_(idxs1, idxs2)]

            if z1 == z2:
                # For same species, the matrix is symmetric and diagonal is 0.
                # We want to count neighbors, so we take values > 0.
                # Since it's symmetric, taking the whole matrix (minus diagonal) counts each bond twice,
                # effectively counting neighbors for *each* atom in idxs1.
                dists = sub_mat[sub_mat > 1e-6]
            else:
                # For different species, take all distances
                dists = sub_mat.flatten()

            if len(dists) > 0:
                hist, _ = np.histogram(dists, bins=bins)
                # Normalize by the number of 'center' atoms (idxs1) to make it intensive
                # This gives "average number of z2 neighbors at distance r around a z1 atom"
                hist = hist.astype(float) / len(idxs1)
            else:
                hist = np.zeros(n_bins)

        # Add to features
        label = f"rdf_{z1}_{z2}"
        for b in range(n_bins):
            histograms[f"{label}_bin_{b}"] = hist[b]

    return histograms


def _compute_angles(vectors):
    """
    Helper to compute angles (in degrees) between all pairs of vectors in the list.
    vectors: (N, 3) array
    """
    n = len(vectors)
    if n < 2:
        return np.array([])

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1)
    # Filter zero-length vectors (shouldn't happen with cutoff > 0)
    valid_mask = norms > 1e-6
    vecs = vectors[valid_mask]
    norms = norms[valid_mask]

    if len(vecs) < 2:
        return np.array([])

    u_vecs = vecs / norms[:, np.newaxis]

    # Compute cosine similarity matrix: (N, 3) @ (3, N) -> (N, N)
    cos_theta = np.dot(u_vecs, u_vecs.T)

    # Clip for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Take upper triangle indices (k > i) to get unique pairs
    triu_idx = np.triu_indices(len(u_vecs), k=1)
    valid_cosines = cos_theta[triu_idx]

    angles = np.degrees(np.arccos(valid_cosines))
    return angles


def compute_adf(atoms: Atoms, r_cut: float, n_bins: int) -> dict:
    """
    Computes Angular Distribution Functions (ADF) for:
    1. Intra-polyhedral: O-Metal-O (Center: Metal, Neighbors: Oxygen)
    2. Inter-polyhedral: Metal-O-Metal (Center: Oxygen, Neighbors: Metal)
    """
    bins = np.linspace(ADF_MIN_ANGLE, ADF_MAX_ANGLE, n_bins + 1)

    hist_intra = np.zeros(n_bins)  # O-Metal-O
    hist_inter = np.zeros(n_bins)  # Metal-O-Metal

    atomic_numbers = atoms.get_atomic_numbers()
    metal_indices = [i for i, z in enumerate(atomic_numbers) if z in METALS]
    oxygen_indices = [i for i, z in enumerate(atomic_numbers) if z == OXYGEN]

    # 1. Intra-polyhedral (Center = Metal)
    if len(metal_indices) > 0 and len(oxygen_indices) > 0:
        for m_idx in metal_indices:
            # Get vectors from metal to all oxygens
            # get_distances returns (distances, vectors) when vector=True
            dists, vecs = atoms.get_distances(
                m_idx, oxygen_indices, mic=True, vector=True
            )

            # Filter by cutoff
            mask = dists < r_cut
            valid_vecs = vecs[mask]  # Vectors from Metal to Oxygen

            angles = _compute_angles(valid_vecs)
            if len(angles) > 0:
                h, _ = np.histogram(angles, bins=bins)
                hist_intra += h

        # Normalize by number of metal centers
        hist_intra /= len(metal_indices)

    # 2. Inter-polyhedral (Center = Oxygen)
    if len(oxygen_indices) > 0 and len(metal_indices) > 0:
        for o_idx in oxygen_indices:
            # Get vectors from oxygen to all metals
            dists, vecs = atoms.get_distances(
                o_idx, metal_indices, mic=True, vector=True
            )

            mask = dists < r_cut
            valid_vecs = vecs[mask]  # Vectors from Oxygen to Metal

            angles = _compute_angles(valid_vecs)
            if len(angles) > 0:
                h, _ = np.histogram(angles, bins=bins)
                hist_inter += h

        # Normalize by number of oxygen centers
        hist_inter /= len(oxygen_indices)

    # Pack into dictionary
    features = {}
    for b in range(n_bins):
        features[f"adf_intra_bin_{b}"] = hist_intra[b]
    for b in range(n_bins):
        features[f"adf_inter_bin_{b}"] = hist_inter[b]

    return features


def process_single_structure(file_path: str) -> dict:
    """
    Worker function to process one structure file.
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    if not os.path.exists(full_path):
        return {}

    try:
        # Fix: Explicitly specify format='aims' because files have .xyz extension but FHI-aims content
        atoms = ase.io.read(full_path, format="aims")

        # 1. Physical Properties
        feats = get_physical_properties(atoms)

        # 2. RDF
        rdf_feats = compute_rdf(atoms, RDF_MIN_DIST, RDF_MAX_DIST, RDF_BINS)
        feats.update(rdf_feats)

        # 3. ADF
        adf_feats = compute_adf(atoms, CUTOFF_DISTANCE, ADF_BINS)
        feats.update(adf_feats)

        return feats
    except Exception as e:
        # Return empty dict on failure, will be handled by filling NaNs later if needed
        print(f"Error processing {file_path}: {e}")
        return {}


def extract_features(
    metadata_df: pd.DataFrame, load_cached_data: bool = True, cache_path: str = None
) -> pd.DataFrame:
    """
    Main feature extraction pipeline.
    Iterates over the metadata DataFrame, loads geometry, computes features,
    and merges them with the tabular metadata.
    """
    # 1. Check Cache
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Computing structural features (RDF + ADF + Physical)...")

    # 2. Process all structures
    features_list = []
    ids = []

    for idx, row in metadata_df.iterrows():
        f_path = row["file_path"]
        feats = process_single_structure(f_path)
        if feats:
            feats["id"] = row["id"]
            features_list.append(feats)
            ids.append(row["id"])

    # 3. Create DataFrame
    if not features_list:
        raise ValueError("No features were extracted. Check input data paths.")

    features_df = pd.DataFrame(features_list)
    features_df.set_index("id", inplace=True)

    # 4. Merge with original metadata (excluding file_path to keep it clean)
    # We want to keep the tabular features (percent_atom_*, spacegroup, etc.)
    meta_copy = metadata_df.copy()
    if "file_path" in meta_copy.columns:
        meta_copy.drop(columns=["file_path"], inplace=True)

    meta_copy.set_index("id", inplace=True)

    # Join inner to ensure we only keep rows where feature extraction succeeded
    full_df = meta_copy.join(features_df, how="inner")

    # 5. Save Cache
    if cache_path:
        print(f"Saving features to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        full_df.to_parquet(cache_path)

    return full_df
