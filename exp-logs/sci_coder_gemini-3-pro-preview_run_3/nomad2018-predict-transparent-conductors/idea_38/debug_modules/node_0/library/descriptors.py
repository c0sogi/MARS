import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import (
    INPUT_DIR,
    RDF_CUTOFF,
    RDF_NUM_BINS,
    BOND_CUTOFF,
    BVS_PARAMS,
    CATIONS,
    ANIONS,
    PERCENTILES,
    ALL_ELEMENTS,
)


def calculate_rdf(atoms):
    """
    Calculates element-resolved Radial Distribution Functions (RDF).
    Explicitly handles Metal-Metal and Metal-Anion pairs.
    """
    # Get all pairwise distances within the cutoff
    # i: source atom index, j: target atom index, d: distance
    i_indices, j_indices, dists = neighbor_list("ijd", atoms, RDF_CUTOFF)

    symbols = np.array(atoms.get_chemical_symbols())
    n_atoms = len(atoms)

    # Define bins
    bins = np.linspace(0, RDF_CUTOFF, RDF_NUM_BINS + 1)

    # Define pairs of interest
    pairs_of_interest = []

    # Metal-Anion pairs (e.g., Al-O, Ga-O, In-O)
    for cat in CATIONS:
        for an in ANIONS:
            pairs_of_interest.append(tuple(sorted((cat, an))))

    # Metal-Metal pairs (e.g., Al-Al, Al-Ga, ...)
    for i in range(len(CATIONS)):
        for j in range(i, len(CATIONS)):
            pairs_of_interest.append(tuple(sorted((CATIONS[i], CATIONS[j]))))

    # Unique sorted pairs to avoid duplicates in dictionary keys
    pairs_of_interest = sorted(list(set(pairs_of_interest)))

    rdf_features = {}

    # Pre-compute symbol arrays for masking
    syms_i = symbols[i_indices]
    syms_j = symbols[j_indices]

    for el1, el2 in pairs_of_interest:
        label = f"RDF_{el1}_{el2}"

        # Select pairs matching (el1, el2)
        # neighbor_list returns both (i,j) and (j,i), so we check both directions or enforce order
        # Since we sorted the tuple (el1, el2), we just need to match the sorted symbols of the pair
        # But doing element-wise sort on large arrays is slow.
        # Instead: (sym_i == el1 & sym_j == el2) | (sym_i == el2 & sym_j == el1)

        if el1 == el2:
            mask = (syms_i == el1) & (syms_j == el2)
        else:
            mask = ((syms_i == el1) & (syms_j == el2)) | (
                (syms_i == el2) & (syms_j == el1)
            )

        pair_dists = dists[mask]

        # Compute histogram
        hist, _ = np.histogram(pair_dists, bins=bins)

        # Normalize by total atom count to make it intensive
        if n_atoms > 0:
            hist = hist / n_atoms

        rdf_features[label] = hist

    return rdf_features


def calculate_local_fingerprints(atoms):
    """
    Calculates atom-wise local descriptors:
    1. Bond Valence Sum (BVS)
    2. Effective Coordination Number (ECoN)
    3. Local Anisotropy Index (Vector sum of bond directions)
    """
    # Get neighbors within bonding cutoff
    # D: vector pointing from i to j
    i_indices, j_indices, dists, vectors = neighbor_list("ijdD", atoms, BOND_CUTOFF)

    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()

    # Initialize arrays
    bvs = np.zeros(n_atoms)
    econ = np.zeros(n_atoms)
    aniso = np.zeros(n_atoms)

    # BVS Parameters
    b_param = BVS_PARAMS["b"]

    # Iterate over each atom to compute its local properties
    for k in range(n_atoms):
        # Find neighbors for atom k
        mask = i_indices == k

        if not np.any(mask):
            continue

        k_dists = dists[mask]
        k_vecs = vectors[mask]
        k_neigh_indices = j_indices[mask]

        # --- 1. Bond Valence Sum (BVS) ---
        # Only calculated for Cations bonded to Oxygen (as per params)
        center_sym = symbols[k]
        val_bvs = 0.0

        if center_sym in CATIONS:
            r0 = BVS_PARAMS.get(center_sym, None)
            if r0:
                for idx, d in enumerate(k_dists):
                    neigh_sym = symbols[k_neigh_indices[idx]]
                    if neigh_sym == "O":
                        val_bvs += np.exp((r0 - d) / b_param)

        bvs[k] = val_bvs

        # --- 2. Effective Coordination Number (ECoN) ---
        # Formula: sum exp(1 - (d_i / d_avg)^6)
        if len(k_dists) > 0:
            d_avg = np.mean(k_dists)
            if d_avg > 1e-6:
                terms = np.exp(1 - (k_dists / d_avg) ** 6)
                econ[k] = np.sum(terms)

        # --- 3. Local Anisotropy ---
        # Magnitude of the sum of normalized bond vectors
        if len(k_dists) > 0:
            # Normalize vectors
            norms = np.linalg.norm(k_vecs, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0  # Safety
            unit_vecs = k_vecs / norms

            # Sum unit vectors
            vec_sum = np.sum(unit_vecs, axis=0)

            # Magnitude
            aniso[k] = np.linalg.norm(vec_sum)

    return bvs, econ, aniso


def aggregate_distributions(atoms, bvs, econ, aniso):
    """
    Aggregates atom-wise features into structure-wise features using percentiles,
    grouped by element type.
    """
    features = {}
    symbols = np.array(atoms.get_chemical_symbols())

    # For each element type of interest
    for el in ALL_ELEMENTS:
        mask = symbols == el

        if np.any(mask):
            el_bvs = bvs[mask]
            el_econ = econ[mask]
            el_aniso = aniso[mask]

            for p in PERCENTILES:
                features[f"BVS_{el}_p{p}"] = np.percentile(el_bvs, p)
                features[f"ECoN_{el}_p{p}"] = np.percentile(el_econ, p)
                features[f"Aniso_{el}_p{p}"] = np.percentile(el_aniso, p)
        else:
            # Fill with 0 if element not present
            for p in PERCENTILES:
                features[f"BVS_{el}_p{p}"] = 0.0
                features[f"ECoN_{el}_p{p}"] = 0.0
                features[f"Aniso_{el}_p{p}"] = 0.0

    return features


def process_geometry(file_path):
    """
    Reads a geometry file and computes the full feature vector.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    try:
        atoms = ase.io.read(full_path)
    except Exception:
        # Return empty or None if file read fails
        return None

    # 1. Macroscopic Features
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 1e-6 else 0.0

    features = {
        "volume_per_atom": vol / len(atoms),
        "density": density,
        "packing_fraction": 0.0,  # Placeholder or could implement if needed
    }

    # 2. Local Fingerprints & Aggregation
    bvs, econ, aniso = calculate_local_fingerprints(atoms)
    dist_features = aggregate_distributions(atoms, bvs, econ, aniso)
    features.update(dist_features)

    # 3. RDF Features
    rdf_dict = calculate_rdf(atoms)
    # Flatten RDF dictionary
    for label, hist in rdf_dict.items():
        for i, val in enumerate(hist):
            features[f"{label}_{i}"] = val

    return features


def generate_features(
    metadata_df, load_cached_data=True, cache_path=None, debug_limit=None
):
    """
    Main function to generate features for a dataset.
    Handles caching to parquet files.
    """
    # Check cache
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {len(metadata_df)} samples...")

    results = []

    # Debug limit
    if debug_limit:
        metadata_df = metadata_df.head(debug_limit)

    for idx, row in metadata_df.iterrows():
        feat_dict = process_geometry(row["file_path"])

        if feat_dict is not None:
            feat_dict["id"] = row["id"]
            results.append(feat_dict)

    df_features = pd.DataFrame(results)

    # Save to cache
    if cache_path and not df_features.empty:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving features to {cache_path}")
        df_features.to_parquet(cache_path)

    return df_features
