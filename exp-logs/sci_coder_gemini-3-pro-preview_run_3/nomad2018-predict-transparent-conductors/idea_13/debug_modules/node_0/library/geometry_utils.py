import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import (
    INPUT_DIR,
    RDF_CUTOFF,
    RDF_BINS,
    RDF_PAIRS,
    ADF_CUTOFF,
    ADF_BINS,
    ADF_TRIPLETS,
    METALS,
    ANIONS,
    WORKING_DIR,
)


def get_physical_descriptors(atoms):
    """
    Calculates unit cell volume and mass density.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    # Density in AMU / Angstrom^3
    density = mass / vol if vol > 0 else 0.0

    return {"phys_volume": vol, "phys_density": density}


def compute_rdf(atoms):
    """
    Computes Radial Distribution Functions for specific element pairs.
    """
    features = {}
    n_atoms = len(atoms)

    # Get all distances matrix (N x N) with periodic boundary conditions
    dists_matrix = atoms.get_all_distances(mic=True)

    symbols = np.array(atoms.get_chemical_symbols())

    for el1, el2 in RDF_PAIRS:
        # Find indices
        idxs1 = np.where(symbols == el1)[0]
        idxs2 = np.where(symbols == el2)[0]

        if len(idxs1) == 0 or len(idxs2) == 0:
            # If element not present, histogram is all zeros
            hist = np.zeros(RDF_BINS)
        else:
            # Extract submatrix
            if el1 == el2:
                # For homo-pairs, we get the full matrix subset
                sub_dists = dists_matrix[np.ix_(idxs1, idxs2)]
                d_vals = sub_dists.flatten()
                # Remove self-distances (approx 0.0)
                d_vals = d_vals[d_vals > 1e-6]
            else:
                # For hetero-pairs
                sub_dists = dists_matrix[np.ix_(idxs1, idxs2)]
                d_vals = sub_dists.flatten()

            # Histogram
            hist, _ = np.histogram(d_vals, bins=RDF_BINS, range=(0, RDF_CUTOFF))

        # Normalize by total atom count
        hist = hist.astype(float) / n_atoms

        # Store in dict
        for b in range(RDF_BINS):
            features[f"RDF_{el1}_{el2}_bin_{b}"] = hist[b]

    return features


def compute_aggregated_adf(atoms):
    """
    Computes Angular Distribution Functions for chemically aggregated triplets.
    """
    features = {}
    n_atoms = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())

    # Get neighbor list with distances vectors
    # i: center, j: neighbor, D: vector from j to i (or i to j depending on convention)
    # ASE neighbor_list('D') returns vector pointing from i to j.
    # We use 'D' to get the displacement vectors directly.
    i_indices, j_indices, d_vectors = neighbor_list("ijD", atoms, ADF_CUTOFF)

    # Pre-compute distances for normalization
    d_norms = np.linalg.norm(d_vectors, axis=1)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        d_vectors_normalized = d_vectors / d_norms[:, np.newaxis]

    # Prepare data structure for fast lookup: group neighbors by center atom index 'i'
    # Sort by i_indices to group contiguous blocks
    sort_order = np.argsort(i_indices)
    i_sorted = i_indices[sort_order]
    j_sorted = j_indices[sort_order]
    v_sorted = d_vectors_normalized[sort_order]

    # Find split points for each center atom
    unique_i, split_indices = np.unique(i_sorted, return_index=True)

    # Map center atom index to its range in the sorted arrays
    # center_map[atom_index] = (start_index, end_index)
    center_map = {
        idx: (start, end)
        for idx, start, end in zip(
            unique_i, split_indices, np.append(split_indices[1:], len(i_sorted))
        )
    }

    # Helper to check atom type against definition (Specific Element or Group)
    def is_type(symbol, type_def):
        if type_def == "Metal":
            return symbol in METALS
        elif type_def == "O":
            return symbol in ANIONS
        else:
            return symbol == type_def

    for center_type, n1_type, n2_type in ADF_TRIPLETS:
        all_angles = []

        # Iterate over all atoms as potential centers
        for center_idx in range(n_atoms):
            center_sym = symbols[center_idx]

            # Check if this atom qualifies as the center type
            if not is_type(center_sym, center_type):
                continue

            # If atom has no neighbors, skip
            if center_idx not in center_map:
                continue

            start, end = center_map[center_idx]
            my_neighbors_indices = j_sorted[start:end]
            my_neighbors_vectors = v_sorted[start:end]

            # Identify valid neighbors for slot 1 and slot 2
            valid_n1_indices = []  # indices relative to the local my_neighbors arrays
            valid_n2_indices = []

            for k, n_idx in enumerate(my_neighbors_indices):
                n_sym = symbols[n_idx]
                if is_type(n_sym, n1_type):
                    valid_n1_indices.append(k)
                if is_type(n_sym, n2_type):
                    valid_n2_indices.append(k)

            # Generate pairs and compute angles
            if n1_type == n2_type:
                # Combinations of the same list (order doesn't matter, avoid duplicates)
                idxs = valid_n1_indices
                n_neigh = len(idxs)
                if n_neigh < 2:
                    continue

                for a in range(n_neigh):
                    for b in range(a + 1, n_neigh):
                        idx_a = idxs[a]
                        idx_b = idxs[b]

                        # Calculate angle using dot product
                        dot_prod = np.dot(
                            my_neighbors_vectors[idx_a], my_neighbors_vectors[idx_b]
                        )
                        # Clip for numerical stability
                        dot_prod = np.clip(dot_prod, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(dot_prod))
                        all_angles.append(angle_deg)
            else:
                # Product of two different lists
                if not valid_n1_indices or not valid_n2_indices:
                    continue

                for idx_a in valid_n1_indices:
                    for idx_b in valid_n2_indices:
                        # Ensure we don't use the same atom as both neighbors (unlikely with different types, but good practice)
                        if my_neighbors_indices[idx_a] == my_neighbors_indices[idx_b]:
                            continue

                        dot_prod = np.dot(
                            my_neighbors_vectors[idx_a], my_neighbors_vectors[idx_b]
                        )
                        dot_prod = np.clip(dot_prod, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(dot_prod))
                        all_angles.append(angle_deg)

        # Compute histogram for this triplet type
        hist, _ = np.histogram(all_angles, bins=ADF_BINS, range=(0, 180))

        # Normalize by total atom count
        hist = hist.astype(float) / n_atoms

        # Store in features dict
        triplet_name = f"{center_type}_{n1_type}_{n2_type}"
        for b in range(ADF_BINS):
            features[f"ADF_{triplet_name}_bin_{b}"] = hist[b]

    return features


def process_single_structure(file_path):
    """
    Loads a structure and computes all features.
    """
    try:
        atoms = ase.io.read(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    feats = {}

    # 1. Physical Descriptors
    feats.update(get_physical_descriptors(atoms))

    # 2. Radial Distribution Functions
    feats.update(compute_rdf(atoms))

    # 3. Angular Distribution Functions
    feats.update(compute_aggregated_adf(atoms))

    return feats


def extract_structural_features(
    metadata_df, load_cached_data=True, cache_file_name="features.parquet"
):
    """
    Main function to process a dataset (train/val/test).
    Handles caching to avoid re-computation.
    """
    cache_path = os.path.join(WORKING_DIR, cache_file_name)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            df_features = pd.read_parquet(cache_path)
            # Simple validation: check if length matches
            if len(df_features) == len(metadata_df):
                return df_features
            else:
                print(
                    f"Cache length mismatch ({len(df_features)} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Computing features for {len(metadata_df)} structures...")

    results = []

    # Iterate through metadata
    for idx, row in metadata_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        feats = process_single_structure(full_path)

        if feats is None:
            # If reading failed, we create an empty dict.
            # Later filling with 0s will handle missing keys.
            feats = {}

        # Add ID to ensure alignment and merging safety
        feats["id"] = row["id"]
        results.append(feats)

    # Convert to DataFrame
    df_features = pd.DataFrame(results)

    # Ensure 'id' is integer for merging
    if "id" in df_features.columns:
        df_features["id"] = df_features["id"].astype(int)

    # Fill NaNs (e.g., if a structure had no triplets of a certain type) with 0.0
    df_features = df_features.fillna(0.0)

    # 3. Save Cache
    print(f"Saving features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path)

    return df_features
