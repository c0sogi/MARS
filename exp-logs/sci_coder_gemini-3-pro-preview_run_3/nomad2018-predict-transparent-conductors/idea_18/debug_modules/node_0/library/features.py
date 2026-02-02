import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from ase.data import atomic_numbers, chemical_symbols
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    ATOMIC_NUMBERS,
    SUBLATTICE_METALS,
    SUBLATTICE_ANIONS,
    RDF_CUTOFF,
    RDF_BINS,
    NEIGHBOR_CUTOFF,
    RANDOM_SEED,
)


def load_structure(rel_path):
    """
    Loads an atomic structure from a relative path within the input directory.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        return None
    try:
        atoms = ase.io.read(full_path)
        return atoms
    except Exception:
        return None


def compute_global_descriptors(atoms):
    """
    Computes global physical descriptors: Volume and Density.
    """
    if atoms is None:
        return {"global_volume": np.nan, "global_density": np.nan}

    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 1e-6 else 0.0

    return {"global_volume": vol, "global_density": density}


def compute_rdf(atoms):
    """
    Computes element-resolved Radial Distribution Functions (RDF).
    Returns a flattened dictionary of histogram bins for relevant pairs.
    """
    if atoms is None:
        return {}

    # Define pairs of interest based on dataset composition (Al, Ga, In, O)
    species = sorted(list(ATOMIC_NUMBERS.keys()))
    pairs = []
    for i in range(len(species)):
        for j in range(i, len(species)):
            pairs.append((species[i], species[j]))

    # Get all distances
    # We use get_all_distances with mic=True to account for periodic boundaries
    # However, for RDF, neighbor_list is often faster/easier for cutoffs
    i_indices, j_indices, dists = neighbor_list("ijd", atoms, RDF_CUTOFF)

    # Map indices to symbols
    symbols = np.array(atoms.get_chemical_symbols())

    rdf_features = {}

    # Pre-calculate bins
    bins = np.linspace(0, RDF_CUTOFF, RDF_BINS + 1)

    # Total volume for normalization (density)
    vol = atoms.get_volume()

    for el1, el2 in pairs:
        # Mask for specific pair
        # neighbor_list returns i, j. We need pairs where (sym[i] == el1 and sym[j] == el2)
        # or (sym[i] == el2 and sym[j] == el1)

        mask = (symbols[i_indices] == el1) & (symbols[j_indices] == el2)

        # If el1 == el2, we might double count or not depending on implementation.
        # neighbor_list returns both i-j and j-i.
        # For histograms, we just take all relevant distances.

        pair_dists = dists[mask]

        hist, _ = np.histogram(pair_dists, bins=bins)

        # Normalize by number of atoms to make it intensive-like
        norm_factor = len(atoms)
        if norm_factor > 0:
            hist = hist / norm_factor

        for k, val in enumerate(hist):
            rdf_features[f"rdf_{el1}_{el2}_{k}"] = val

    return rdf_features


def compute_local_moments(atoms):
    """
    Computes local geometric moments for each atom:
    - Coordination Number (CN)
    - Bond Length Mean and Variance
    - Bond Angle Mean and Variance

    Returns a DataFrame where each row corresponds to an atom in the structure.
    """
    if atoms is None:
        return pd.DataFrame()

    n_atoms = len(atoms)
    indices = range(n_atoms)

    # Get neighbors
    i_idx, j_idx, d_ij = neighbor_list("ijd", atoms, NEIGHBOR_CUTOFF)
    # Also get vector distances for angles
    # D_ij vector points from i to j
    D_ij = neighbor_list("D", atoms, NEIGHBOR_CUTOFF)

    # Initialize storage
    cns = np.zeros(n_atoms)
    bl_means = np.full(n_atoms, np.nan)
    bl_vars = np.full(n_atoms, np.nan)
    ba_means = np.full(n_atoms, np.nan)
    ba_vars = np.full(n_atoms, np.nan)

    # Process per atom
    # We can iterate because n_atoms is small (~80 max)
    for i in indices:
        # Neighbors of i
        mask = i_idx == i
        if not np.any(mask):
            cns[i] = 0
            continue

        dists = d_ij[mask]
        vecs = D_ij[mask]

        # Coordination Number
        cns[i] = len(dists)

        # Bond Length Statistics
        bl_means[i] = np.mean(dists)
        bl_vars[i] = np.var(dists) if len(dists) > 1 else 0.0

        # Bond Angles
        # Calculate angles between all pairs of neighbors
        # Cosine rule: a.b / |a||b|
        if len(dists) > 1:
            angles = []
            n_neigh = len(dists)
            for a in range(n_neigh):
                for b in range(a + 1, n_neigh):
                    v1 = vecs[a]
                    v2 = vecs[b]
                    d1 = dists[a]
                    d2 = dists[b]

                    dot_prod = np.dot(v1, v2)
                    cosine = dot_prod / (d1 * d2)
                    # Clip for numerical stability
                    cosine = np.clip(cosine, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cosine))
                    angles.append(angle)

            if angles:
                ba_means[i] = np.mean(angles)
                ba_vars[i] = np.var(angles)
            else:
                ba_means[i] = 0.0
                ba_vars[i] = 0.0
        else:
            ba_means[i] = 0.0
            ba_vars[i] = 0.0

    df_local = pd.DataFrame(
        {
            "CN": cns,
            "BL_mean": bl_means,
            "BL_var": bl_vars,
            "BA_mean": ba_means,
            "BA_var": ba_vars,
        }
    )

    return df_local


def aggregate_hierarchical_features(local_stats_df, symbols):
    """
    Aggregates local moments at two levels:
    1. Element-specific (Al, Ga, In, O)
    2. Sublattice-specific (Metal, Anion)
    """
    if local_stats_df.empty:
        return {}

    # Add symbol column for grouping
    local_stats_df["symbol"] = symbols

    # Define sublattices
    # Metal: Al, Ga, In
    # Anion: O
    def get_sublattice(sym):
        if sym in SUBLATTICE_METALS:
            return "Metal"
        elif sym in SUBLATTICE_ANIONS:
            return "Anion"
        else:
            return "Other"

    local_stats_df["sublattice"] = local_stats_df["symbol"].apply(get_sublattice)

    agg_features = {}

    # Metrics to aggregate
    metrics = ["CN", "BL_mean", "BL_var", "BA_mean", "BA_var"]

    # Level 1: Element-specific aggregation
    for el in ATOMIC_NUMBERS.keys():
        subset = local_stats_df[local_stats_df["symbol"] == el]
        if not subset.empty:
            for m in metrics:
                agg_features[f"el_{el}_{m}_mean"] = subset[m].mean()
                agg_features[f"el_{el}_{m}_std"] = subset[m].std(
                    ddof=0
                )  # Population std or sample std, 0 is fine
        else:
            # Fill with NaN or 0? XGBoost handles NaN.
            for m in metrics:
                agg_features[f"el_{el}_{m}_mean"] = np.nan
                agg_features[f"el_{el}_{m}_std"] = np.nan

    # Level 2: Sublattice-specific aggregation
    for sub in ["Metal", "Anion"]:
        subset = local_stats_df[local_stats_df["sublattice"] == sub]
        if not subset.empty:
            for m in metrics:
                agg_features[f"sub_{sub}_{m}_mean"] = subset[m].mean()
                agg_features[f"sub_{sub}_{m}_std"] = subset[m].std(ddof=0)
        else:
            for m in metrics:
                agg_features[f"sub_{sub}_{m}_mean"] = np.nan
                agg_features[f"sub_{sub}_{m}_std"] = np.nan

    return agg_features


def process_dataset(metadata_path, output_path, load_cached_data=True):
    """
    Main processing function.
    Reads metadata, computes features (Global, RDF, Hierarchical Moments),
    merges with tabular data, and caches the result.
    """
    # 1. Check cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}...")
        return pd.read_parquet(output_path)

    print(f"Processing dataset from {metadata_path}...")

    # 2. Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # 3. Iterate and compute features
    # We collect list of dicts to create DataFrame later
    feature_rows = []

    for idx, row in df.iterrows():
        # Load atoms
        atoms = load_structure(row["file_path"])

        if atoms is None:
            # Handle missing file case (though metadata validation says 0 missing)
            # Create a row of NaNs for features
            feature_rows.append({})
            continue

        # A. Global
        global_feats = compute_global_descriptors(atoms)

        # B. RDF
        rdf_feats = compute_rdf(atoms)

        # C. Hierarchical Local Moments
        local_df = compute_local_moments(atoms)
        symbols = atoms.get_chemical_symbols()
        hier_feats = aggregate_hierarchical_features(local_df, symbols)

        # Combine all
        combined = {**global_feats, **rdf_feats, **hier_feats}
        feature_rows.append(combined)

    # 4. Create DataFrame from features
    feat_df = pd.DataFrame(feature_rows)

    # 5. Merge with original metadata (excluding file_path if not needed, but keeping ID is good)
    # We concatenate horizontally. Ensure indices align (they should).
    result_df = pd.concat([df, feat_df], axis=1)

    # 6. Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    print(f"Saved processed features to {output_path}")

    return result_df
