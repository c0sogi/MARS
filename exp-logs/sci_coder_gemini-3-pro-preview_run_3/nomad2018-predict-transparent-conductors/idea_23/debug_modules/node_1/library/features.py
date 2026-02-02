import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    RDF_CUTOFF,
    RDF_NUM_BINS,
    BOND_CUTOFF,
    PERCENTILES,
)


def extract_physical_properties(atoms):
    """
    Calculates unit cell volume and mass density from an ASE Atoms object.

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Dictionary containing 'volume' and 'density'.
    """
    vol = atoms.get_volume()
    # ASE masses are in atomic mass units (u). Volume is in Angstrom^3.
    # Density here is in u/A^3.
    total_mass = sum(atoms.get_masses())
    density = total_mass / vol if vol > 0 else 0.0
    return {"volume": vol, "density": density}


def extract_rdf(atoms):
    """
    Computes element-resolved Radial Distribution Functions (RDF) histograms.

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Flattened dictionary of RDF histogram bins for each element pair.
    """
    # Define expected elements to ensure consistent feature space
    expected_elements = ["Al", "Ga", "In", "O"]

    rdf_features = {}

    # Use neighbor_list to efficiently find all pairs within RDF_CUTOFF
    # 'd' returns distances
    i_indices, j_indices, d_indices = neighbor_list("ijd", atoms, RDF_CUTOFF)

    symbols = np.array(atoms.get_chemical_symbols())

    # Define histogram bins
    bins = np.linspace(0, RDF_CUTOFF, RDF_NUM_BINS + 1)

    # Iterate through all unique pairs of elements
    for i, el1 in enumerate(expected_elements):
        for j, el2 in enumerate(expected_elements):
            if j < i:
                continue

            pair_label = f"rdf_{el1}_{el2}"

            # Create masks for the current element pair
            # neighbor_list returns both i->j and j->i, so we need to be careful not to double count
            # or we can just normalize appropriately.
            # Here we select all i-j edges where symbol[i] == el1 and symbol[j] == el2.

            mask_i = symbols[i_indices] == el1
            mask_j = symbols[j_indices] == el2

            if el1 == el2:
                # For same-element pairs, mask_i and mask_j are identical.
                mask = mask_i & mask_j
            else:
                # For different elements, we want distances for Al-O (which covers O-Al in the neighbor list if we check both)
                # neighbor_list contains both directions. We can just sum up the occurrences.
                # Let's capture both A->B and B->A to ensure we get the full distribution.
                mask = (mask_i & mask_j) | (
                    (symbols[i_indices] == el2) & (symbols[j_indices] == el1)
                )

            dists = d_indices[mask]

            # Compute histogram
            hist, _ = np.histogram(dists, bins=bins)

            # Normalize by the number of atoms in the cell to make the descriptor intensive
            norm_hist = hist / len(atoms)

            # Populate dictionary
            for k, val in enumerate(norm_hist):
                rdf_features[f"{pair_label}_bin_{k}"] = val

    return rdf_features


def extract_interaction_fingerprints(atoms):
    """
    Extracts distributional fingerprints (percentiles) of bond lengths and bond angles
    for chemically resolved interactions.

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Dictionary of percentiles for specific bond types and angle types.
    """
    # 1. Neighbor Analysis with Bond Cutoff
    # 'i', 'j' are indices, 'd' is distance, 'D' is distance vector (r_j - r_i)
    i_indices, j_indices, d_indices, D_vectors = neighbor_list(
        "ijdD", atoms, BOND_CUTOFF
    )

    symbols = np.array(atoms.get_chemical_symbols())

    # Dictionaries to collect populations of geometric values
    # Key: tuple(sorted(el1, el2)), Value: list of lengths
    bond_lengths_pop = {}
    # Key: string "Angle_Neighbor1_Center_Neighbor2", Value: list of angles (degrees)
    bond_angles_pop = {}

    # --- Collect Bond Lengths ---
    for k in range(len(i_indices)):
        idx_i = i_indices[k]
        idx_j = j_indices[k]
        dist = d_indices[k]

        sym_i = symbols[idx_i]
        sym_j = symbols[idx_j]

        # Sort symbols to ensure Al-O and O-Al map to the same key
        pair_key = tuple(sorted((sym_i, sym_j)))
        pair_str = f"bond_{pair_key[0]}_{pair_key[1]}"

        if pair_str not in bond_lengths_pop:
            bond_lengths_pop[pair_str] = []
        bond_lengths_pop[pair_str].append(dist)

    # --- Collect Bond Angles ---
    # Build an adjacency list for easier angle iteration: center_idx -> list of (neighbor_idx, vector, distance)
    adj_list = [[] for _ in range(len(atoms))]
    for k in range(len(i_indices)):
        # D_vectors[k] points from i to j
        adj_list[i_indices[k]].append((j_indices[k], D_vectors[k], d_indices[k]))

    for center_idx, neighbors in enumerate(adj_list):
        center_sym = symbols[center_idx]
        n_neighbors = len(neighbors)

        # Need at least 2 neighbors to form an angle
        if n_neighbors < 2:
            continue

        # Iterate over unique pairs of neighbors attached to the center
        for n1 in range(n_neighbors):
            for n2 in range(n1 + 1, n_neighbors):
                idx1, vec1, dist1 = neighbors[n1]
                idx2, vec2, dist2 = neighbors[n2]

                sym1 = symbols[idx1]
                sym2 = symbols[idx2]

                # Calculate angle using dot product
                # cos(theta) = (v1 . v2) / (|v1| |v2|)
                dot_prod = np.dot(vec1, vec2)
                denominator = dist1 * dist2

                # Avoid division by zero (though dists should be > 0)
                if denominator < 1e-8:
                    continue

                cos_theta = np.clip(dot_prod / denominator, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))

                # Construct key: Neighbor1-Center-Neighbor2
                # Sort neighbors to ensure symmetry (e.g. O-Al-O)
                neigh_syms = sorted((sym1, sym2))
                angle_key_str = f"angle_{neigh_key[0]}_{center_sym}_{neigh_key[1]}"

                if angle_key_str not in bond_angles_pop:
                    bond_angles_pop[angle_key_str] = []
                bond_angles_pop[angle_key_str].append(angle_deg)

    # --- Compute Percentiles ---
    features = {}

    def compute_stats(name, values):
        res = {}
        if not values:
            return res

        # Calculate percentiles
        try:
            pct_values = np.percentile(values, PERCENTILES)
            for p, val in zip(PERCENTILES, pct_values):
                res[f"{name}_p{p}"] = val
        except Exception:
            pass
        return res

    # Process Bonds
    for name, values in bond_lengths_pop.items():
        features.update(compute_stats(name, values))

    # Process Angles
    for name, values in bond_angles_pop.items():
        features.update(compute_stats(name, values))

    return features


def process_data(metadata_path, load_cached_data=True):
    """
    Main function to process geometry files listed in metadata and return a DataFrame of features.
    Implements caching to parquet files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing IDs, targets (if available), and extracted features.
    """
    # Construct cache path based on metadata filename
    base_name = os.path.basename(metadata_path).replace(".csv", "_features.parquet")
    cache_path = os.path.join(WORKING_DIR, base_name)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute from Scratch
    print(f"Processing geometry data from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    features_list = []

    # Iterate through each material in the metadata
    for idx, row in df_meta.iterrows():
        # Construct full path to geometry file
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        try:
            atoms = ase.io.read(full_path, format="aims")
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            continue

        # Initialize feature dictionary with ID and Metadata info
        feat_row = {
            "id": row["id"],
            "spacegroup": row["spacegroup"],
            "percent_atom_al": row["percent_atom_al"],
            "percent_atom_ga": row["percent_atom_ga"],
            "percent_atom_in": row["percent_atom_in"],
        }

        # Add targets if they exist (train/val sets)
        for target in ["formation_energy_ev_natom", "bandgap_energy_ev"]:
            if target in row:
                feat_row[target] = row[target]

        # Extract Geometric Features
        feat_row.update(extract_physical_properties(atoms))
        feat_row.update(extract_rdf(atoms))
        feat_row.update(extract_interaction_fingerprints(atoms))

        features_list.append(feat_row)

    # Create DataFrame
    df_features = pd.DataFrame(features_list)

    # 3. Save to Cache
    print(f"Saving features to {cache_path}")
    # Ensure directory exists (redundant if config does it, but safe)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path, index=False)

    return df_features
