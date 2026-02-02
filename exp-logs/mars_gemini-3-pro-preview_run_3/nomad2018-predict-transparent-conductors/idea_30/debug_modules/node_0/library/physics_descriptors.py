import os
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from ase.geometry import get_distances
from scipy.spatial.distance import pdist, squareform

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    BVS_PARAMS,
    FORMAL_VALENCES,
    RDF_CUTOFF,
    RDF_BINS,
    ANGLE_CUTOFF,
    RANDOM_SEED,
)

# Ensure reproducible results
np.random.seed(RANDOM_SEED)


def get_pbc_vectors(atoms: Atoms, cutoff: float):
    """
    Computes vectors and distances between all atoms respecting PBC.
    Returns:
        vectors: (N, N, 3) array where vectors[i, j] is vector from i to j
        distances: (N, N) array of distances
    """
    # This uses ASE's internal neighbor list logic or get_distances which handles MIC
    # get_distances returns (positions, distances)
    # We need vectors.
    # A robust way for dense dense matrices in small cells is using get_distances(mic=True, vector=True)

    # mic=True applies Minimum Image Convention
    # vector=True returns the vector r_ij
    rij_vectors = atoms.get_all_distances(mic=True, vector=True)
    distances = np.linalg.norm(rij_vectors, axis=2)

    return rij_vectors, distances


def calculate_bond_valences_and_vectors(
    atoms: Atoms, rij_vectors: np.ndarray, distances: np.ndarray
):
    """
    Calculates Scalar BVS and Vector BVVS for each atom.
    Only considers Metal-Oxygen bonds.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    n_atoms = len(atoms)

    # Initialize arrays
    scalar_bvs = np.zeros(n_atoms)
    vector_bvs = np.zeros((n_atoms, 3))

    # Universal softness parameter
    b = BVS_PARAMS["b"]

    # Iterate through all pairs
    # We can optimize by masking, but loops are clear for logic
    # Since N < 100 typically, N^2 is small.

    for i in range(n_atoms):
        elem_i = symbols[i]

        for j in range(n_atoms):
            if i == j:
                continue

            dist = distances[i, j]
            if dist > 6.0:  # Optimization: BVS is negligible at large distances
                continue

            elem_j = symbols[j]

            # Check if pair is Metal-Oxygen
            # Case 1: i is Metal, j is Oxygen
            r0 = None
            if elem_i in ["Al", "Ga", "In"] and elem_j == "O":
                r0 = BVS_PARAMS[elem_i]
            # Case 2: i is Oxygen, j is Metal
            elif elem_i == "O" and elem_j in ["Al", "Ga", "In"]:
                r0 = BVS_PARAMS[elem_j]

            if r0 is not None:
                # Calculate bond valence
                s_ij = np.exp((r0 - dist) / b)

                # Add to scalar sum
                scalar_bvs[i] += s_ij

                # Add to vector sum
                # Vector points from i to j.
                # Normalize vector r_ij
                if dist > 1e-6:
                    unit_vec = rij_vectors[i, j] / dist
                    vector_bvs[i] += s_ij * unit_vec

    # Compute magnitude of resultant vector
    bvvs_magnitude = np.linalg.norm(vector_bvs, axis=1)

    return scalar_bvs, bvvs_magnitude


def calculate_gii(atoms: Atoms, scalar_bvs: np.ndarray):
    """
    Calculates Global Instability Index (GII).
    GII = sqrt( sum( (Vi - V_ideal)^2 ) / N )
    """
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    squared_diff_sum = 0.0

    for i, sym in enumerate(symbols):
        if sym in FORMAL_VALENCES:
            v_ideal = FORMAL_VALENCES[sym]
            diff = scalar_bvs[i] - v_ideal
            squared_diff_sum += diff * diff
        else:
            # Should not happen in this dataset
            pass

    gii = np.sqrt(squared_diff_sum / n_atoms)
    return gii


def compute_rdf(atoms: Atoms, distances: np.ndarray):
    """
    Computes Radial Distribution Function features for specific pairs.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    n_atoms = len(atoms)

    # Define pairs of interest
    pairs_of_interest = [
        ("Al", "O"),
        ("Ga", "O"),
        ("In", "O"),
        ("O", "O"),
        ("Al", "Al"),
        ("Ga", "Ga"),
        ("In", "In"),
    ]

    rdf_features = {}

    # Pre-compute bins
    bins = np.linspace(0, RDF_CUTOFF, RDF_BINS + 1)

    for el1, el2 in pairs_of_interest:
        # Find indices
        idxs1 = np.where(symbols == el1)[0]
        idxs2 = np.where(symbols == el2)[0]

        if len(idxs1) == 0 or len(idxs2) == 0:
            # If element missing, fill with zeros
            hist = np.zeros(RDF_BINS)
        else:
            # Extract submatrix of distances
            # If el1 == el2, we need upper triangle or handle i != j
            # The distances matrix is full.

            # We want all pairs between set 1 and set 2
            # If sets are same, avoid double counting?
            # Standard RDF counts all neighbors j for each i.

            # Let's collect all relevant distances
            relevant_dists = []
            for i in idxs1:
                for j in idxs2:
                    if i == j:
                        continue
                    d = distances[i, j]
                    if d <= RDF_CUTOFF:
                        relevant_dists.append(d)

            if not relevant_dists:
                hist = np.zeros(RDF_BINS)
            else:
                hist, _ = np.histogram(relevant_dists, bins=bins)
                # Normalize by number of atoms of type 1 (density independent)
                hist = hist / len(idxs1)

        # Store features
        for b in range(RDF_BINS):
            rdf_features[f"rdf_{el1}_{el2}_bin_{b}"] = hist[b]

    return rdf_features


def compute_angles(atoms: Atoms, rij_vectors: np.ndarray, distances: np.ndarray):
    """
    Computes distribution of Metal-Oxygen-Metal angles.
    """
    symbols = np.array(atoms.get_chemical_symbols())

    # Indices
    o_indices = np.where(symbols == "O")[0]
    metal_indices = np.where(np.isin(symbols, ["Al", "Ga", "In"]))[0]

    angles = []

    for o_idx in o_indices:
        # Find bonded metals within cutoff
        # Using a slightly larger cutoff than typical bond length to catch all
        # typical M-O bonds are ~1.9-2.2 A. Cutoff 3.0 is safe.

        # Get neighbors
        # We look at row o_idx in distances
        dists = distances[o_idx, metal_indices]

        # Filter by cutoff
        valid_mask = dists < ANGLE_CUTOFF
        valid_metals = metal_indices[valid_mask]

        if len(valid_metals) < 2:
            continue

        # Calculate angles for all pairs of metals connected to this oxygen
        # Vector from O to M is needed.
        # rij_vectors[o_idx, m_idx] is vector from O to M

        for i in range(len(valid_metals)):
            m1 = valid_metals[i]
            v1 = rij_vectors[o_idx, m1]
            d1 = distances[o_idx, m1]

            for j in range(i + 1, len(valid_metals)):
                m2 = valid_metals[j]
                v2 = rij_vectors[o_idx, m2]
                d2 = distances[o_idx, m2]

                # Dot product
                dot_prod = np.dot(v1, v2)
                cosine = dot_prod / (d1 * d2)

                # Clip for numerical stability
                cosine = np.clip(cosine, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cosine))
                angles.append(angle_deg)

    if not angles:
        return {
            "angle_min": 0,
            "angle_max": 0,
            "angle_mean": 0,
            "angle_std": 0,
            "angle_25": 0,
            "angle_50": 0,
            "angle_75": 0,
        }

    return {
        "angle_min": np.min(angles),
        "angle_max": np.max(angles),
        "angle_mean": np.mean(angles),
        "angle_std": np.std(angles),
        "angle_25": np.percentile(angles, 25),
        "angle_50": np.median(angles),
        "angle_75": np.percentile(angles, 75),
    }


def extract_features_for_structure(file_path: str):
    """
    Main extraction logic for a single structure file.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    try:
        atoms = ase.io.read(full_path)
    except Exception:
        # Return zeros if file read fails (unlikely)
        return None

    # 1. Global Physical
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 0 else 0.0
    n_atoms = len(atoms)

    # 2. Distances and Vectors (PBC)
    rij_vectors, distances = get_pbc_vectors(
        atoms, cutoff=10.0
    )  # cutoff not strictly used by get_all_distances but good for concept

    # 3. Bond Valence Analysis
    scalar_bvs, bvvs_mag = calculate_bond_valences_and_vectors(
        atoms, rij_vectors, distances
    )
    gii = calculate_gii(atoms, scalar_bvs)

    # Aggregations for BVS and BVVS
    symbols = np.array(atoms.get_chemical_symbols())
    bvs_stats = {}

    for el in ["Al", "Ga", "In", "O"]:
        mask = symbols == el
        if np.any(mask):
            vals_s = scalar_bvs[mask]
            vals_v = bvvs_mag[mask]

            bvs_stats[f"bvs_{el}_mean"] = np.mean(vals_s)
            bvs_stats[f"bvs_{el}_std"] = np.std(vals_s)
            bvs_stats[f"bvs_{el}_max"] = np.max(vals_s)
            bvs_stats[f"bvs_{el}_min"] = np.min(vals_s)

            bvs_stats[f"bvvs_{el}_mean"] = np.mean(vals_v)
            bvs_stats[f"bvvs_{el}_max"] = np.max(vals_v)
        else:
            bvs_stats[f"bvs_{el}_mean"] = 0.0
            bvs_stats[f"bvs_{el}_std"] = 0.0
            bvs_stats[f"bvs_{el}_max"] = 0.0
            bvs_stats[f"bvs_{el}_min"] = 0.0

            bvs_stats[f"bvvs_{el}_mean"] = 0.0
            bvs_stats[f"bvvs_{el}_max"] = 0.0

    # 4. RDF
    rdf_feats = compute_rdf(atoms, distances)

    # 5. Angles
    angle_feats = compute_angles(atoms, rij_vectors, distances)

    # Combine all
    features = {
        "vol_per_atom": vol / n_atoms,
        "density": density,
        "gii": gii,
        **bvs_stats,
        **rdf_feats,
        **angle_feats,
    }

    return features


def generate_features(
    metadata_df: pd.DataFrame, split: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Generates or loads features for the given metadata DataFrame.

    Args:
        metadata_df: DataFrame containing 'file_path' and 'id'.
        split: 'train', 'val', or 'test'.
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        DataFrame with extracted features.
    """

    cache_path = os.path.join(WORKING_DIR, f"{split}_features.parquet")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {split} set ({len(metadata_df)} samples)...")

    feature_list = []
    ids = []

    for idx, row in metadata_df.iterrows():
        feats = extract_features_for_structure(row["file_path"])
        if feats is not None:
            feature_list.append(feats)
            ids.append(row["id"])

    # Create DataFrame
    feat_df = pd.DataFrame(feature_list)
    feat_df["id"] = ids

    # Merge with original metadata to keep targets and other tabular info if needed
    # But usually we just return features. Let's return features + id.
    # The caller can merge.

    # Save to cache
    print(f"Saving features to {cache_path}")
    feat_df.to_parquet(cache_path, index=False)

    return feat_df
