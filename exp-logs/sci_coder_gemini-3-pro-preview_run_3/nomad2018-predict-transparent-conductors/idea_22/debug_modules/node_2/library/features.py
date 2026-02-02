import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from itertools import combinations_with_replacement
from library.config import (
    ATOM_LIST,
    BOND_CUTOFF,
    RDF_CUTOFF,
    RDF_NUM_BINS,
    PERCENTILES,
    WORK_DIR,
    RANDOM_SEED,
)
from library.data_loader import load_geometry


def compute_physical_descriptors(atoms):
    """
    Computes basic physical descriptors: Volume and Density.
    """
    vol = atoms.get_volume()
    masses = atoms.get_masses()
    total_mass = np.sum(masses)
    # Density in AMU / Angstrom^3
    density = total_mass / vol if vol > 1e-6 else 0.0

    return {"vol_ang3": vol, "density_amu_ang3": density}


def compute_rdf(atoms):
    """
    Computes element-resolved Radial Distribution Functions (RDF).
    """
    # Define bins
    bins = np.linspace(0, RDF_CUTOFF, RDF_NUM_BINS + 1)

    # Get all pairwise distances within cutoff
    i_indices, j_indices, dists = neighbor_list("ijd", atoms, RDF_CUTOFF)

    chemical_symbols = np.array(atoms.get_chemical_symbols())

    # Initialize dictionary for RDF features
    rdf_features = {}

    # Generate all pairs
    pairs = list(combinations_with_replacement(sorted(ATOM_LIST), 2))

    # Pre-calculate histograms for each pair
    for el1, el2 in pairs:
        sym_i = chemical_symbols[i_indices]
        sym_j = chemical_symbols[j_indices]

        # Mask for pairs (el1, el2) considering symmetry
        if el1 == el2:
            mask = (sym_i == el1) & (sym_j == el2)
        else:
            mask = ((sym_i == el1) & (sym_j == el2)) | ((sym_i == el2) & (sym_j == el1))

        pair_dists = dists[mask]

        # Compute histogram
        hist, _ = np.histogram(pair_dists, bins=bins)

        # Normalize by total number of atoms
        norm_hist = hist / len(atoms)

        # Store in dict
        for k, count in enumerate(norm_hist):
            rdf_features[f"RDF_{el1}_{el2}_{k}"] = count

    return rdf_features


def compute_cation_fingerprints(atoms):
    """
    Computes distributional fingerprints for Cations (Al, Ga, In).
    Metrics: Coordination Number (CN) and Bond Angle Variance.
    """
    # Get neighbors within bonding cutoff
    i_indices, j_indices, D_vectors = neighbor_list("ijD", atoms, BOND_CUTOFF)

    chemical_symbols = np.array(atoms.get_chemical_symbols())

    # Prepare storage
    features = {}

    # Define Cations
    cations = ["Al", "Ga", "In"]

    # Sort by i for efficient grouping
    sort_order = np.argsort(i_indices)
    i_sorted = i_indices[sort_order]
    D_sorted = D_vectors[sort_order]

    # Find unique indices and their counts/locations
    unique_i, split_indices = np.unique(i_sorted, return_index=True)

    # Initialize lists to hold stats for each element type
    stats_per_element = {el: {"cn": [], "var": []} for el in cations}

    # Map atom index to its element
    atom_elements = atoms.get_chemical_symbols()

    # Helper to calculate angle variance
    def calc_angle_variance(vectors):
        n = len(vectors)
        if n < 2:
            return 0.0

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-9
        if np.sum(valid) < 2:
            return 0.0

        vecs = vectors[valid] / norms[valid][:, np.newaxis]

        # Compute dot products for all pairs
        dots = np.dot(vecs, vecs.T)
        dots = np.clip(dots, -1.0, 1.0)

        # Angles in degrees
        angles = np.degrees(np.arccos(dots))

        # We only want upper triangle off-diagonal (unique pairs)
        r, c = np.triu_indices(len(vecs), k=1)
        if len(r) == 0:
            return 0.0

        unique_angles = angles[r, c]
        return np.var(unique_angles)

    # Map from atom_idx to (start, end) in D_sorted
    atom_neighbor_ranges = {}
    for k, start in enumerate(split_indices):
        end = split_indices[k + 1] if k + 1 < len(split_indices) else len(i_sorted)
        atom_idx = unique_i[k]
        atom_neighbor_ranges[atom_idx] = (start, end)

    for atom_idx, element in enumerate(atom_elements):
        if element not in cations:
            continue

        if atom_idx in atom_neighbor_ranges:
            start, end = atom_neighbor_ranges[atom_idx]
            vectors = D_sorted[start:end]
            cn = len(vectors)
            var = calc_angle_variance(vectors)
        else:
            cn = 0
            var = 0.0

        stats_per_element[element]["cn"].append(cn)
        stats_per_element[element]["var"].append(var)

    # Aggregate into percentiles
    for el in cations:
        cns = stats_per_element[el]["cn"]
        vars_ = stats_per_element[el]["var"]

        for p in PERCENTILES:
            # CN features
            if len(cns) > 0:
                val_cn = np.percentile(cns, p)
            else:
                val_cn = 0.0
            features[f"{el}_CN_p{p}"] = val_cn

            # Variance features
            if len(vars_) > 0:
                val_var = np.percentile(vars_, p)
            else:
                val_var = 0.0
            features[f"{el}_Var_p{p}"] = val_var

    return features


def compute_anion_fingerprints(atoms):
    """
    Computes distributional fingerprints for Anions (Oxygen).
    Metric: Metal-Oxygen-Metal (M-O-M) bridging angles.
    """
    # Get neighbors within bonding cutoff
    i_indices, j_indices, D_vectors = neighbor_list("ijD", atoms, BOND_CUTOFF)

    atom_elements = np.array(atoms.get_chemical_symbols())

    # Identify O indices
    o_indices_in_atoms = [idx for idx, el in enumerate(atom_elements) if el == "O"]

    # Identify Metal indices
    metal_set = set(["Al", "Ga", "In"])

    # Sort neighbor list by center atom i
    sort_order = np.argsort(i_indices)
    i_sorted = i_indices[sort_order]
    j_sorted = j_indices[sort_order]
    D_sorted = D_vectors[sort_order]

    unique_i, split_indices = np.unique(i_sorted, return_index=True)

    atom_neighbor_ranges = {}
    for k, start in enumerate(split_indices):
        end = split_indices[k + 1] if k + 1 < len(split_indices) else len(i_sorted)
        atom_idx = unique_i[k]
        atom_neighbor_ranges[atom_idx] = (start, end)

    all_mom_angles = []

    for o_idx in o_indices_in_atoms:
        if o_idx not in atom_neighbor_ranges:
            continue

        start, end = atom_neighbor_ranges[o_idx]

        # Get neighbors
        neigh_indices = j_sorted[start:end]
        neigh_vectors = D_sorted[start:end]

        # Filter for metal neighbors
        metal_mask = [atom_elements[n] in metal_set for n in neigh_indices]
        metal_vectors = neigh_vectors[metal_mask]

        if len(metal_vectors) < 2:
            continue

        # Calculate angles between all pairs of metal neighbors
        norms = np.linalg.norm(metal_vectors, axis=1)
        valid = norms > 1e-9
        if np.sum(valid) < 2:
            continue

        vecs = metal_vectors[valid] / norms[valid][:, np.newaxis]

        # Dot products
        dots = np.dot(vecs, vecs.T)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))

        # Upper triangle
        r, c = np.triu_indices(len(vecs), k=1)
        if len(r) > 0:
            unique_angles = angles[r, c]
            all_mom_angles.extend(unique_angles)

    # Compute percentiles of the global distribution
    features = {}
    for p in PERCENTILES:
        if len(all_mom_angles) > 0:
            val = np.percentile(all_mom_angles, p)
        else:
            val = 0.0
        features[f"Anion_MOM_Angle_p{p}"] = val

    return features


def generate_feature_vector(atoms):
    """
    Orchestrates the feature extraction for a single atomic structure.
    """
    # 1. Physical Descriptors
    phys_feats = compute_physical_descriptors(atoms)

    # 2. Radial Fingerprints
    rdf_feats = compute_rdf(atoms)

    # 3. Cation Fingerprints
    cat_feats = compute_cation_fingerprints(atoms)

    # 4. Anion Fingerprints
    an_feats = compute_anion_fingerprints(atoms)

    # Combine all
    features = {**phys_feats, **rdf_feats, **cat_feats, **an_feats}

    return features


def process_data(df, load_cached_data=True):
    """
    Processes the dataframe to generate features.
    Implements caching mechanism.

    Args:
        df (pd.DataFrame): Dataframe containing 'file_path' and metadata.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Dataframe with generated features and original metadata.
    """
    # Create unique cache filename based on IDs
    ids_hash = pd.util.hash_pandas_object(df[["id"]]).sum()
    cache_path = os.path.join(WORK_DIR, f"features_{ids_hash}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Computing features from scratch...")

    feature_list = []

    # Iterate over rows
    for idx, row in df.iterrows():
        try:
            atoms = load_geometry(row["file_path"])
            feats = generate_feature_vector(atoms)
            # Add ID to ensure alignment
            feats["id"] = row["id"]
            feature_list.append(feats)
        except Exception as e:
            print(f"Error processing ID {row['id']}: {e}")
            pass

    # Create features DataFrame
    features_df = pd.DataFrame(feature_list)

    # Merge with original metadata
    if "id" in features_df.columns:
        result_df = df.merge(features_df, on="id", how="left")
    else:
        result_df = df.copy()

    # Save to cache
    try:
        result_df.to_parquet(cache_path, index=False)
        print(f"Saved features to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return result_df
