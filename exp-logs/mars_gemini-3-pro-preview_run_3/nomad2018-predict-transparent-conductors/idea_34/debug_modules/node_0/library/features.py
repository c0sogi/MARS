import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

# Import configuration
from library.config import (
    R0_VALUES,
    PERCENTILES,
    RDF_R_MIN,
    RDF_R_MAX,
    RDF_N_BINS,
    ANGLE_N_BINS,
    ANGLE_MIN,
    ANGLE_MAX,
    INPUT_DIR,
    WORKING_DIR,
    RANDOM_SEED,
)

# Constants for calculations
BVS_B = 0.37  # Empirical constant for Bond Valence Sum
ANGLE_CUTOFF = 3.0  # Angstroms, cutoff for defining bonds for angle calculation
RDF_BIN_EDGES = np.linspace(RDF_R_MIN, RDF_R_MAX, RDF_N_BINS + 1)
ANGLE_BIN_EDGES = np.linspace(ANGLE_MIN, ANGLE_MAX, ANGLE_N_BINS + 1)


def compute_bvs(atoms, neighbor_indices, distances, chemical_symbols):
    """
    Computes Bond Valence Sums (BVS) for each atom.
    Only considers Metal-Oxygen bonds.
    """
    n_atoms = len(atoms)
    bvs_values = np.zeros(n_atoms)

    # Iterate over neighbor pairs
    for i, j, d in zip(neighbor_indices[0], neighbor_indices[1], distances):
        elem_i = chemical_symbols[i]
        elem_j = chemical_symbols[j]

        # We only care about Metal-Oxygen interactions
        # Check if one is Metal and other is Oxygen
        if elem_i == "O" and elem_j in R0_VALUES:
            r0 = R0_VALUES[elem_j]
            val = np.exp((r0 - d) / BVS_B)
            bvs_values[i] += val
        elif elem_i in R0_VALUES and elem_j == "O":
            r0 = R0_VALUES[elem_i]
            val = np.exp((r0 - d) / BVS_B)
            bvs_values[i] += val

    return bvs_values


def compute_econ(atoms, neighbor_indices, distances):
    """
    Computes Effective Coordination Number (ECoN) using Hoppe's definition.
    ECoN_i = sum_j exp(1 - (d_ij / d_min_i)^6)
    """
    n_atoms = len(atoms)
    econ_values = np.zeros(n_atoms)

    # Find min distance for each atom
    min_dists = np.full(n_atoms, np.inf)
    for i, d in zip(neighbor_indices[0], distances):
        if d < min_dists[i]:
            min_dists[i] = d

    # Compute ECoN
    for i, d in zip(neighbor_indices[0], distances):
        if min_dists[i] > 0:  # Avoid division by zero
            term = np.exp(1.0 - (d / min_dists[i]) ** 6)
            econ_values[i] += term

    return econ_values


def compute_rdf(atoms, neighbor_indices, distances, chemical_symbols):
    """
    Computes Radial Distribution Functions for specific pairs.
    Returns histograms normalized by number of atoms.
    """
    rdfs = {}
    n_atoms = len(atoms)
    volume = atoms.get_volume()

    # Define pairs of interest
    pairs_of_interest = [
        ("Al", "O"),
        ("Ga", "O"),
        ("In", "O"),
        ("Metal", "Metal"),  # Aggregated cation-cation
    ]

    # Initialize histograms
    for p1, p2 in pairs_of_interest:
        rdfs[f"RDF_{p1}_{p2}"] = np.zeros(RDF_N_BINS)

    # Categorize interactions
    for i, j, d in zip(neighbor_indices[0], neighbor_indices[1], distances):
        # Avoid double counting if iterating full list (i,j) and (j,i)
        # neighbor_list returns both. We can just process all and normalize later.

        elem_i = chemical_symbols[i]
        elem_j = chemical_symbols[j]

        # Metal-Oxygen
        if elem_i in R0_VALUES and elem_j == "O":
            key = f"RDF_{elem_i}_O"
            bin_idx = int((d - RDF_R_MIN) / (RDF_R_MAX - RDF_R_MIN) * RDF_N_BINS)
            if 0 <= bin_idx < RDF_N_BINS:
                rdfs[key][bin_idx] += 1.0

        # Metal-Metal
        if elem_i in R0_VALUES and elem_j in R0_VALUES:
            key = "RDF_Metal_Metal"
            bin_idx = int((d - RDF_R_MIN) / (RDF_R_MAX - RDF_R_MIN) * RDF_N_BINS)
            if 0 <= bin_idx < RDF_N_BINS:
                rdfs[key][bin_idx] += 1.0

    # Flatten to features
    features = {}
    for key, hist in rdfs.items():
        # Normalize by number of atoms to make it intensive
        hist /= n_atoms
        for b in range(RDF_N_BINS):
            features[f"{key}_bin_{b}"] = hist[b]

    return features


def compute_angles(atoms, chemical_symbols):
    """
    Computes bond angle distributions for O-M-O and M-O-M.
    Uses a fixed cutoff to define neighbors for angle calculation.
    """
    cutoff = ANGLE_CUTOFF
    # Get neighbors for angle calculation
    # We need indices of neighbors for each atom
    i_idx, j_idx, d_ij, vectors = neighbor_list("ijdD", atoms, cutoff)

    # Organize neighbors by atom index
    neighbors = {i: [] for i in range(len(atoms))}
    for idx, (i, j, vec) in enumerate(zip(i_idx, j_idx, vectors)):
        neighbors[i].append((j, vec))

    omo_angles = []
    mom_angles = []

    for i in range(len(atoms)):
        elem_center = chemical_symbols[i]
        nbs = neighbors[i]
        if len(nbs) < 2:
            continue

        # Check center type
        is_metal = elem_center in R0_VALUES
        is_oxygen = elem_center == "O"

        if not (is_metal or is_oxygen):
            continue

        # Iterate pairs of neighbors to find angles
        # We only care about O-M-O (Center=Metal, Neighbors=Oxygen)
        # and M-O-M (Center=Oxygen, Neighbors=Metal)

        for idx1 in range(len(nbs)):
            for idx2 in range(idx1 + 1, len(nbs)):
                j1, v1 = nbs[idx1]
                j2, v2 = nbs[idx2]

                elem1 = chemical_symbols[j1]
                elem2 = chemical_symbols[j2]

                # Calculate angle
                # v1 and v2 are vectors from i to j.
                # Angle is arccos( (v1.v2) / (|v1|*|v2|) )
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 < 1e-3 or norm2 < 1e-3:
                    continue

                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                # Clip for numerical stability
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))

                if is_metal and elem1 == "O" and elem2 == "O":
                    omo_angles.append(angle_deg)
                elif is_oxygen and elem1 in R0_VALUES and elem2 in R0_VALUES:
                    mom_angles.append(angle_deg)

    # Compute stats
    features = {}

    for name, angles in [("Angle_OMO", omo_angles), ("Angle_MOM", mom_angles)]:
        if len(angles) > 0:
            for p in PERCENTILES:
                features[f"{name}_p{p}"] = np.percentile(angles, p)
            features[f"{name}_std"] = np.std(angles)
        else:
            for p in PERCENTILES:
                features[f"{name}_p{p}"] = np.nan
            features[f"{name}_std"] = np.nan

    return features


def process_structure(row):
    """
    Process a single structure to extract all features.
    """
    file_path = os.path.join(INPUT_DIR, row["file_path"])

    try:
        atoms = ase.io.read(file_path)
    except Exception:
        # Return empty or NaN features if file read fails
        return None

    chemical_symbols = np.array(atoms.get_chemical_symbols())

    # 1. Macroscopic Features
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 0 else 0.0
    n_atoms = len(atoms)

    features = {
        "volume_per_atom": vol / n_atoms,
        "density": density,
        "packing_fraction": 0.0,  # Placeholder, ECoN covers this
    }

    # 2. Neighbor List for BVS, ECoN, RDF
    # Use a generous cutoff for RDF, but filter for BVS/ECoN
    cutoff = RDF_R_MAX
    i_idx, j_idx, d_ij = neighbor_list("ijd", atoms, cutoff)

    # 3. BVS & ECoN (Atomic Scale)
    # We need a smaller cutoff for these, effectively the first coordination shell.
    # Typically < 3.0-4.0 A for these oxides. Let's filter the large neighbor list.
    mask_local = d_ij < 4.0
    i_local = i_idx[mask_local]
    j_local = j_idx[mask_local]
    d_local = d_ij[mask_local]

    bvs_all = compute_bvs(atoms, (i_local, j_local), d_local, chemical_symbols)
    econ_all = compute_econ(atoms, (i_local, j_local), d_local)

    # Aggregation by element
    elements = ["Al", "Ga", "In", "O"]
    for elem in elements:
        mask_elem = chemical_symbols == elem
        if np.any(mask_elem):
            bvs_elem = bvs_all[mask_elem]
            econ_elem = econ_all[mask_elem]

            for p in PERCENTILES:
                features[f"BVS_{elem}_p{p}"] = np.percentile(bvs_elem, p)
                features[f"ECoN_{elem}_p{p}"] = np.percentile(econ_elem, p)
        else:
            for p in PERCENTILES:
                features[f"BVS_{elem}_p{p}"] = np.nan
                features[f"ECoN_{elem}_p{p}"] = np.nan

    # Aggregate for all Cations
    mask_cation = np.isin(chemical_symbols, ["Al", "Ga", "In"])
    if np.any(mask_cation):
        bvs_cat = bvs_all[mask_cation]
        econ_cat = econ_all[mask_cation]
        for p in PERCENTILES:
            features[f"BVS_Cation_p{p}"] = np.percentile(bvs_cat, p)
            features[f"ECoN_Cation_p{p}"] = np.percentile(econ_cat, p)
    else:
        for p in PERCENTILES:
            features[f"BVS_Cation_p{p}"] = np.nan
            features[f"ECoN_Cation_p{p}"] = np.nan

    # 4. RDF (Interaction Scale)
    # Uses the full cutoff neighbor list
    rdf_feats = compute_rdf(atoms, (i_idx, j_idx), d_ij, chemical_symbols)
    features.update(rdf_feats)

    # 5. Angles (Topology)
    # Re-computes neighbors internally with ANGLE_CUTOFF
    angle_feats = compute_angles(atoms, chemical_symbols)
    features.update(angle_feats)

    return features


def generate_features(df, split_name, load_cached_data=True):
    """
    Main function to generate features for a given metadata DataFrame.
    Handles caching and parallel processing.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'file_path'.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Feature matrix.
    """
    cache_path = os.path.join(WORKING_DIR, f"{split_name}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {split_name} set ({len(df)} samples)...")

    # Convert dataframe rows to list of dicts for parallel processing
    rows = df.to_dict("records")

    # Parallel processing
    # n_jobs=-1 uses all available cores
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_structure)(row) for row in rows
    )

    # Filter out None results (failed reads)
    # We maintain alignment by index. If read failed, we might have rows with NaNs.
    # Ideally, we should have a feature row for every input row.
    # If result is None, replace with empty dict (will become NaNs)
    results = [res if res is not None else {} for res in results]

    feat_df = pd.DataFrame(results)

    # Ensure index matches input df
    feat_df.index = df.index

    # Save to cache
    print(f"Saving features to {cache_path}")
    feat_df.to_parquet(cache_path)

    return feat_df
