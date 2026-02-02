import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from ase.neighborlist import neighbor_list
import library.config as config
from library.data_loader import load_metadata, get_data_generator

# =============================================================================
# Constants for Physics Calculations
# =============================================================================
# Bond Valence Parameters (R0 in Angstroms, B is typically 0.37)
# Source: Brown, I. D. & Altermatt, D. (1985). Acta Cryst. B41, 244-247.
BVS_PARAMS = {"Al": 1.651, "Ga": 1.730, "In": 1.900}
BVS_B = 0.37

# Cutoff for defining coordination environment (M-O bonds)
# Typically slightly larger than the first coordination shell
COORDINATION_CUTOFF = 2.8

# =============================================================================
# Helper Functions
# =============================================================================


def compute_percentiles(values, prefix):
    """
    Computes percentiles for a list of values and returns a dictionary.
    If values is empty, returns NaNs.
    """
    features = {}
    if len(values) == 0:
        for p in config.PERCENTILES:
            features[f"{prefix}_p{p}"] = np.nan
        features[f"{prefix}_mean"] = np.nan
        features[f"{prefix}_std"] = np.nan
    else:
        for p in config.PERCENTILES:
            features[f"{prefix}_p{p}"] = np.percentile(values, p)
        features[f"{prefix}_mean"] = np.mean(values)
        features[f"{prefix}_std"] = np.std(values)
    return features


# =============================================================================
# Feature Extraction Functions
# =============================================================================


def extract_macroscopic(atoms):
    """
    Extracts macroscopic properties: Volume, Density, Composition.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 0 else 0.0

    # Composition
    chemical_symbols = atoms.get_chemical_symbols()
    n_atoms = len(chemical_symbols)
    comp = {el: chemical_symbols.count(el) / n_atoms for el in config.ELEMENTS}

    features = {
        "vol_per_atom": vol / n_atoms,
        "density": density,
        "composition_Al": comp.get("Al", 0.0),
        "composition_Ga": comp.get("Ga", 0.0),
        "composition_In": comp.get("In", 0.0),
        "composition_O": comp.get("O", 0.0),
    }
    return features


def extract_rdf(atoms):
    """
    Computes Element-Resolved Radial Distribution Functions.
    Focuses on Metal-Oxygen and Metal-Metal pairs.
    """
    features = {}

    # Get all pairwise distances with mic (minimum image convention)
    # Note: For large cells/datasets, rigorous RDF requires supercells.
    # Here we use a simplified approach suitable for ML descriptors within the unit cell
    # or slightly extended, but ASE's get_all_distances helps.
    # However, get_all_distances(mic=True) is computationally expensive for many atoms.
    # We will use neighbor_list for efficiency with a cutoff.

    # Define pairs of interest
    pairs = []
    # Metal-Oxygen
    for m in config.METALS:
        pairs.append(tuple(sorted((m, "O"))))
    # Metal-Metal
    for i in range(len(config.METALS)):
        for j in range(i, len(config.METALS)):
            pairs.append(tuple(sorted((config.METALS[i], config.METALS[j]))))

    # Initialize histograms
    bins = np.linspace(0, config.RDF_CUTOFF, config.RDF_NUM_BINS + 1)
    histograms = {p: np.zeros(config.RDF_NUM_BINS) for p in pairs}

    # Use neighbor list to get distances
    # i: index of atom 1, j: index of atom 2, d: distance
    i_indices, j_indices, d_values = neighbor_list("ijd", atoms, config.RDF_CUTOFF)

    symbols = np.array(atoms.get_chemical_symbols())

    # Iterate through neighbors and fill histograms
    # This loop can be slow in pure python, but dataset size is manageable.
    # Optimization: Vectorize by grouping indices.

    for p in pairs:
        el1, el2 = p
        # Find indices where (atom_i is el1 AND atom_j is el2) OR (atom_i is el2 AND atom_j is el1)
        # Note: neighbor_list returns both i-j and j-i. We need to be careful not to double count
        # if we normalize, but for raw feature vector, consistent counting is key.

        mask = (symbols[i_indices] == el1) & (symbols[j_indices] == el2)
        if el1 != el2:
            mask |= (symbols[i_indices] == el2) & (symbols[j_indices] == el1)

        d_p = d_values[mask]

        if len(d_p) > 0:
            hist, _ = np.histogram(d_p, bins=bins)
            # Normalize by volume and number of atoms to make it intensive
            # A simple normalization is dividing by total atoms
            hist = hist / len(atoms)
            histograms[p] = hist

    # Flatten histograms into features
    for p, hist in histograms.items():
        pair_name = f"rdf_{p[0]}_{p[1]}"
        # We can't use all bins as features (too many), so we aggregate or take peaks.
        # Let's take simple statistics of the distribution of distances (mean, std)
        # and maybe peak height/location if possible.
        # Alternatively, binning is standard. Let's use coarse binning (e.g. 6 bins)
        # or just statistical moments of the distances found.
        # Let's use moments of the distances found within cutoff.

        # Re-extract distances for this pair to compute moments
        el1, el2 = p
        mask = (symbols[i_indices] == el1) & (symbols[j_indices] == el2)
        if el1 != el2:
            mask |= (symbols[i_indices] == el2) & (symbols[j_indices] == el1)
        d_p = d_values[mask]

        if len(d_p) > 0:
            features[f"{pair_name}_mean"] = np.mean(d_p)
            features[f"{pair_name}_std"] = np.std(d_p)
            features[f"{pair_name}_min"] = np.min(d_p)
            features[f"{pair_name}_count"] = len(d_p) / len(atoms)  # Coordination-like
        else:
            features[f"{pair_name}_mean"] = np.nan
            features[f"{pair_name}_std"] = np.nan
            features[f"{pair_name}_min"] = np.nan
            features[f"{pair_name}_count"] = 0.0

    return features


def extract_site_metrics(atoms):
    """
    Calculates Bond Valence Sums (BVS) and Coordination Numbers (CN) for cations.
    Aggregates them into percentiles.
    """
    symbols = np.array(atoms.get_chemical_symbols())

    # Get neighbors for BVS calculation (Metal-Oxygen only)
    i_indices, j_indices, d_values = neighbor_list("ijd", atoms, COORDINATION_CUTOFF)

    # Initialize arrays
    n_atoms = len(atoms)
    bvs_values = np.zeros(n_atoms)
    cn_values = np.zeros(n_atoms)

    # Compute BVS and CN
    # We iterate over the neighbor list arrays
    for k in range(len(d_values)):
        idx_i = i_indices[k]
        idx_j = j_indices[k]
        dist = d_values[k]
        sym_i = symbols[idx_i]
        sym_j = symbols[idx_j]

        # Only consider Metal-Oxygen bonds
        if sym_i in config.METALS and sym_j == "O":
            r0 = BVS_PARAMS[sym_i]
            val = np.exp((r0 - dist) / BVS_B)
            bvs_values[idx_i] += val
            cn_values[idx_i] += 1
        elif sym_i == "O" and sym_j in config.METALS:
            # For Oxygen, we sum contributions from metals
            r0 = BVS_PARAMS[sym_j]
            val = np.exp((r0 - dist) / BVS_B)
            bvs_values[idx_i] += val  # Valence received
            cn_values[idx_i] += 1

    # Aggregate by element type
    features = {}
    for el in config.ELEMENTS:
        indices = np.where(symbols == el)[0]
        if len(indices) > 0:
            el_bvs = bvs_values[indices]
            el_cn = cn_values[indices]

            # BVS features
            bvs_feats = compute_percentiles(el_bvs, f"bvs_{el}")
            features.update(bvs_feats)

            # CN features
            cn_feats = compute_percentiles(el_cn, f"cn_{el}")
            features.update(cn_feats)
        else:
            # Fill NaNs if element not present
            features.update(compute_percentiles([], f"bvs_{el}"))
            features.update(compute_percentiles([], f"cn_{el}"))

    return features


def extract_alloy_topology(atoms):
    """
    Analyzes Metal-Oxygen-Metal (M-O-M) bond angles.
    Classifies them by metal pair types (e.g., Al-O-Al, Al-O-Ga).
    """
    # We need connectivity. We reuse neighbor list.
    # Cutoff: COORDINATION_CUTOFF
    # We want to find O atoms, then look at their Metal neighbors.

    # Get neighbor list with vectors to calculate angles
    # i: center (O), j: neighbor (M)
    # We need vector D_ij
    cutoff = COORDINATION_CUTOFF
    nl = neighbor_list("ijD", atoms, cutoff)
    i_indices, j_indices, D_vectors = nl

    symbols = np.array(atoms.get_chemical_symbols())

    # Dictionary to store angles for each pair type
    # Key: tuple sorted metal symbols, e.g., ('Al', 'Ga')
    angle_collections = {}
    # Initialize possible pairs
    for i in range(len(config.METALS)):
        for j in range(i, len(config.METALS)):
            pair = tuple(sorted((config.METALS[i], config.METALS[j])))
            angle_collections[pair] = []

    # Iterate over all Oxygen atoms
    o_indices = np.where(symbols == "O")[0]

    for o_idx in o_indices:
        # Find neighbors of this oxygen
        # mask for where i == o_idx
        mask = i_indices == o_idx
        neighbors = j_indices[mask]
        vectors = D_vectors[mask]

        # Filter for Metal neighbors only
        metal_neighbors = []
        metal_vectors = []

        for n_idx, vec in zip(neighbors, vectors):
            if symbols[n_idx] in config.METALS:
                metal_neighbors.append(symbols[n_idx])
                metal_vectors.append(vec)

        # If less than 2 metal neighbors, no angle
        if len(metal_neighbors) < 2:
            continue

        # Calculate angles for all unique pairs of neighbors
        for idx1 in range(len(metal_neighbors)):
            for idx2 in range(idx1 + 1, len(metal_neighbors)):
                m1 = metal_neighbors[idx1]
                m2 = metal_neighbors[idx2]
                v1 = metal_vectors[idx1]
                v2 = metal_vectors[idx2]

                # Calculate angle
                # dot product = |v1||v2| cos(theta)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 1e-6 and norm2 > 1e-6:
                    dot_prod = np.dot(v1, v2)
                    cos_theta = dot_prod / (norm1 * norm2)
                    # Clip for numerical stability
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_theta))

                    pair_key = tuple(sorted((m1, m2)))
                    angle_collections[pair_key].append(angle_deg)

    # Aggregate features
    features = {}
    for pair, angles in angle_collections.items():
        prefix = f"angle_{pair[0]}_{pair[1]}"
        feats = compute_percentiles(angles, prefix)
        features.update(feats)

    return features


def generate_features(df):
    """
    Main feature generation loop.
    """
    all_features = []

    data_gen = get_data_generator(df)

    for idx, row, atoms in data_gen:
        # 1. Macroscopic
        macro = extract_macroscopic(atoms)

        # 2. RDF
        rdf = extract_rdf(atoms)

        # 3. Site Metrics (BVS, CN)
        site = extract_site_metrics(atoms)

        # 4. Alloy Topology (Angles)
        topo = extract_alloy_topology(atoms)

        # Combine
        combined = {**macro, **rdf, **site, **topo}

        # Add ID for reference
        combined["id"] = row["id"]

        # Add target variables if they exist (train/val)
        for target in ["formation_energy_ev_natom", "bandgap_energy_ev"]:
            if target in row:
                combined[target] = row[target]

        all_features.append(combined)

    return pd.DataFrame(all_features)


def process_data(split, load_cached_data=True):
    """
    Main entry point for feature processing.
    Handles loading metadata, checking cache, generating features, and saving cache.
    """
    # 1. Define Cache Path
    cache_filename = f"{split}_features_idea35"
    if config.DEBUG:
        cache_filename += f"_debug_{config.DEBUG_SAMPLE_SIZE}"
    cache_filename += ".parquet"
    cache_path = os.path.join(config.WORKING_DIR, cache_filename)

    # 2. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing...")

    # 3. Load Metadata
    print(f"Processing {split} data...")
    df_meta = load_metadata(split, load_cached_data=True)

    # 4. Generate Features
    df_features = generate_features(df_meta)

    # 5. Save Cache
    print(f"Saving {split} features to cache: {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
