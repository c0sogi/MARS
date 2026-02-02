import os
import numpy as np
import pandas as pd
from ase import Atoms

# Patch for numpy 1.24+ compatibility with older libraries
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
from ase.neighborlist import NeighborList, neighbor_list
from scipy.spatial.distance import pdist, squareform

# Constants
CACHE_DIR = "./working/idea_37"
R_CUT_RDF = 6.0
R_CUT_BOND = 3.0
RDF_BINS = 60
BVS_PARAMS = {
    ("Al", "O"): 1.651,
    ("Ga", "O"): 1.720,
    ("In", "O"): 1.905,
    ("O", "Al"): 1.651,
    ("O", "Ga"): 1.720,
    ("O", "In"): 1.905,
}
B_PARAM = 0.37


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def calculate_macroscopic_props(atoms: Atoms) -> dict:
    """Calculates global properties like volume and density."""
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 0 else 0.0

    # Lattice parameters
    cell = atoms.get_cell_lengths_and_angles()

    return {
        "vol_per_atom": vol / len(atoms),
        "density": density,
        "lattice_a": cell[0],
        "lattice_b": cell[1],
        "lattice_c": cell[2],
        "angle_alpha": cell[3],
        "angle_beta": cell[4],
        "angle_gamma": cell[5],
    }


def calculate_elemental_rdfs(atoms: Atoms, r_max=R_CUT_RDF, n_bins=RDF_BINS) -> dict:
    """Calculates element-resolved Radial Distribution Functions."""
    elements = ["Al", "Ga", "In", "O"]
    pairs = []
    for i in range(len(elements)):
        for j in range(i, len(elements)):
            pairs.append(tuple(sorted((elements[i], elements[j]))))

    # Get all distances using neighbor_list for PBC handling
    i_indices, j_indices, d = neighbor_list("ijd", atoms, r_max)

    symbols = np.array(atoms.get_chemical_symbols())

    rdf_features = {}

    # Pre-calculate bins
    bins = np.linspace(0, r_max, n_bins + 1)

    # Calculate shell volumes for normalization
    r_inner = bins[:-1]
    r_outer = bins[1:]
    v_shell = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)

    for el1, el2 in pairs:
        # Create mask for current pair type
        # Check both (i=el1, j=el2) and (i=el2, j=el1)
        mask1 = (symbols[i_indices] == el1) & (symbols[j_indices] == el2)
        if el1 != el2:
            mask2 = (symbols[i_indices] == el2) & (symbols[j_indices] == el1)
            mask = mask1 | mask2
        else:
            mask = mask1

        d_pair = d[mask]

        if len(d_pair) == 0:
            hist = np.zeros(n_bins)
        else:
            hist, _ = np.histogram(d_pair, bins=bins)

        # Normalize by shell volume and total number of atoms
        # This makes the feature intensive and density-independent
        norm_hist = hist / (v_shell * len(atoms))

        for k, val in enumerate(norm_hist):
            rdf_features[f"rdf_{el1}_{el2}_{k}"] = val

    return rdf_features


def calculate_local_environment(atoms: Atoms, cutoff=R_CUT_BOND) -> dict:
    """Calculates atom-centric features: Bond Valence Sums and Coordination Numbers."""
    i_indices, j_indices, d = neighbor_list("ijd", atoms, cutoff)
    symbols = atoms.get_chemical_symbols()

    bvs_values = np.zeros(len(atoms))
    cn_values = np.zeros(len(atoms))

    for k, (i, j, dist) in enumerate(zip(i_indices, j_indices, d)):
        el_i = symbols[i]
        el_j = symbols[j]

        # Coordination Number (simple count within cutoff)
        cn_values[i] += 1

        # Bond Valence Sum
        if (el_i, el_j) in BVS_PARAMS:
            r0 = BVS_PARAMS[(el_i, el_j)]
            val = np.exp((r0 - dist) / B_PARAM)
            bvs_values[i] += val

    # Aggregate by element type using percentiles
    features = {}
    elements = ["Al", "Ga", "In", "O"]
    percentiles = [0, 25, 50, 75, 100]

    for el in elements:
        indices = [i for i, s in enumerate(symbols) if s == el]
        if not indices:
            for p in percentiles:
                features[f"bvs_{el}_p{p}"] = np.nan
                features[f"cn_{el}_p{p}"] = np.nan
        else:
            el_bvs = bvs_values[indices]
            el_cn = cn_values[indices]

            p_bvs = np.percentile(el_bvs, percentiles)
            p_cn = np.percentile(el_cn, percentiles)

            for p, val in zip(percentiles, p_bvs):
                features[f"bvs_{el}_p{p}"] = val
            for p, val in zip(percentiles, p_cn):
                features[f"cn_{el}_p{p}"] = val

    return features


def calculate_interaction_distributions(atoms: Atoms, cutoff=R_CUT_BOND) -> dict:
    """Calculates interaction-centric features: Bond Valences and Bond Angles distributions."""
    # Use ASE NeighborList for efficient neighbor querying
    nl = NeighborList(
        [cutoff / 2.0] * len(atoms), skin=0.0, sorted=False, self_interaction=False
    )
    nl.update(atoms)

    symbols = atoms.get_chemical_symbols()

    # Store lists of properties for aggregation
    bond_valences = {"Al-O": [], "Ga-O": [], "In-O": []}

    angles = {
        "O-Al-O": [],
        "O-Ga-O": [],
        "O-In-O": [],
        "Al-O-Al": [],
        "Al-O-Ga": [],
        "Al-O-In": [],
        "Ga-O-Ga": [],
        "Ga-O-In": [],
        "In-O-In": [],
    }

    # Iterate over atoms to compute local interactions
    for i in range(len(atoms)):
        el_i = symbols[i]
        indices, offsets = nl.get_neighbors(i)

        # --- Bond Valences ---
        # We collect bonds from the Metal perspective to avoid double counting logic complexity
        # (though double counting wouldn't affect percentiles much, cleanliness is preferred)
        if el_i in ["Al", "Ga", "In"]:
            for idx, offset in zip(indices, offsets):
                el_j = symbols[idx]
                if el_j == "O":
                    # Calculate distance
                    d = atoms.get_distance(
                        i, idx, mic=True, vector=np.dot(offset, atoms.get_cell())
                    )

                    # Calculate individual bond valence
                    r0 = BVS_PARAMS.get((el_i, el_j), 0)
                    bv = np.exp((r0 - d) / B_PARAM)
                    bond_valences[f"{el_i}-{el_j}"].append(bv)

        # --- Angles ---
        # 1. O-Metal-O angles (Polyhedral angles)
        if el_i in ["Al", "Ga", "In"]:
            if len(indices) >= 2:
                # Filter for Oxygen neighbors
                o_neighbors = []
                for idx, offset in zip(indices, offsets):
                    if symbols[idx] == "O":
                        # Vector from central atom i to neighbor
                        vec = (
                            atoms.positions[idx]
                            + np.dot(offset, atoms.get_cell())
                            - atoms.positions[i]
                        )
                        o_neighbors.append(vec)

                # Compute angles for all unique pairs of O neighbors
                for a in range(len(o_neighbors)):
                    for b in range(a + 1, len(o_neighbors)):
                        v1 = o_neighbors[a]
                        v2 = o_neighbors[b]
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 > 1e-3 and norm2 > 1e-3:
                            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(cos_theta))
                            angles[f"O-{el_i}-O"].append(angle_deg)

        # 2. Metal-O-Metal angles (Bridging angles)
        if el_i == "O":
            if len(indices) >= 2:
                # Filter for Metal neighbors
                m_neighbors = []
                m_symbols = []
                for idx, offset in zip(indices, offsets):
                    sym = symbols[idx]
                    if sym in ["Al", "Ga", "In"]:
                        vec = (
                            atoms.positions[idx]
                            + np.dot(offset, atoms.get_cell())
                            - atoms.positions[i]
                        )
                        m_neighbors.append(vec)
                        m_symbols.append(sym)

                # Compute angles
                for a in range(len(m_neighbors)):
                    for b in range(a + 1, len(m_neighbors)):
                        v1 = m_neighbors[a]
                        v2 = m_neighbors[b]
                        s1 = m_symbols[a]
                        s2 = m_symbols[b]

                        # Sort symbols alphabetically to match dictionary keys
                        pair_key = sorted([s1, s2])
                        key = f"{pair_key[0]}-O-{pair_key[1]}"

                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 > 1e-3 and norm2 > 1e-3:
                            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(cos_theta))
                            if key in angles:
                                angles[key].append(angle_deg)

    # Aggregate distributions into percentiles
    features = {}
    percentiles = [0, 25, 50, 75, 100]

    # Bond Valence Percentiles
    for pair, vals in bond_valences.items():
        if len(vals) > 0:
            stats = np.percentile(vals, percentiles)
            for p, val in zip(percentiles, stats):
                features[f"bv_{pair}_p{p}"] = val
        else:
            for p in percentiles:
                features[f"bv_{pair}_p{p}"] = np.nan

    # Angle Percentiles
    for triple, vals in angles.items():
        if len(vals) > 0:
            stats = np.percentile(vals, percentiles)
            for p, val in zip(percentiles, stats):
                features[f"angle_{triple}_p{p}"] = val
        else:
            for p in percentiles:
                features[f"angle_{triple}_p{p}"] = np.nan

    return features


def process_single_structure(atoms: Atoms) -> dict:
    """Wrapper to run all descriptor calculations for a single structure."""
    feats = {}
    feats.update(calculate_macroscopic_props(atoms))
    feats.update(calculate_elemental_rdfs(atoms))
    feats.update(calculate_local_environment(atoms))
    feats.update(calculate_interaction_distributions(atoms))
    return feats


def get_descriptors(df: pd.DataFrame, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Main function to generate descriptors for a dataframe of structures.
    Handles caching to avoid re-computing expensive descriptors.
    """
    ensure_dir(CACHE_DIR)

    # Create a unique hash for the cache file based on the IDs in the dataframe
    ids_hash = pd.util.hash_pandas_object(df["id"]).sum()
    cache_file = os.path.join(CACHE_DIR, f"features_{ids_hash}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Computing features from scratch...")

    # Import here to avoid circular imports if any, though not expected
    from library.data_handler import load_geometry

    # Load all atoms objects
    atoms_list = load_geometry(df)

    features_list = []
    for i, atoms in enumerate(atoms_list):
        try:
            f = process_single_structure(atoms)
            # Add ID to ensure alignment
            f["id"] = df.iloc[i]["id"]
            features_list.append(f)
        except Exception as e:
            print(f"Error processing structure index {i} (ID {df.iloc[i]['id']}): {e}")
            # Append a placeholder with ID to keep alignment, filled with NaNs
            features_list.append({"id": df.iloc[i]["id"]})

    feat_df = pd.DataFrame(features_list)

    # Save cache
    try:
        feat_df.to_parquet(cache_file)
        print(f"Features saved to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

    return feat_df
