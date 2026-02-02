import os
import numpy as np
import pandas as pd
import ase.io
from library.config import INPUT_DIR, WORKING_DIR, RANDOM_SEED, set_seed
from library.geometry_descriptors import (
    calculate_bond_valences,
    calculate_weighted_bond_angles,
    calculate_structural_metrics,
)

# Set seed for reproducibility
set_seed(RANDOM_SEED)


def weighted_quantile(values, weights, quantiles=0.5):
    """
    Very simple weighted quantile implementation.
    """
    if len(values) == 0:
        return np.full(len(np.atleast_1d(quantiles)), np.nan)

    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return np.interp(np.array(quantiles) * c[-1], c, values[i])


def compute_stats(array, weights=None, prefix=""):
    """
    Computes summary statistics for an array, optionally weighted.
    """
    stats = {}
    if len(array) == 0:
        # Return NaNs for all expected keys
        keys = ["mean", "std", "min", "p25", "p50", "p75", "max", "range"]
        for k in keys:
            stats[f"{prefix}_{k}"] = np.nan
        return stats

    if weights is None:
        # Unweighted stats
        stats[f"{prefix}_mean"] = np.mean(array)
        stats[f"{prefix}_std"] = np.std(array)
        stats[f"{prefix}_min"] = np.min(array)
        stats[f"{prefix}_max"] = np.max(array)
        stats[f"{prefix}_range"] = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]

        percentiles = np.percentile(array, [25, 50, 75])
        stats[f"{prefix}_p25"] = percentiles[0]
        stats[f"{prefix}_p50"] = percentiles[1]
        stats[f"{prefix}_p75"] = percentiles[2]
    else:
        # Weighted stats
        # Normalize weights
        w = weights / np.sum(weights)
        weighted_mean = np.sum(array * w)
        weighted_var = np.sum(w * (array - weighted_mean) ** 2)

        stats[f"{prefix}_mean"] = weighted_mean
        stats[f"{prefix}_std"] = np.sqrt(weighted_var)
        stats[f"{prefix}_min"] = np.min(
            array
        )  # Min/Max don't depend on weights typically
        stats[f"{prefix}_max"] = np.max(array)
        stats[f"{prefix}_range"] = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]

        qs = weighted_quantile(array, weights, quantiles=[0.25, 0.50, 0.75])
        stats[f"{prefix}_p25"] = qs[0]
        stats[f"{prefix}_p50"] = qs[1]
        stats[f"{prefix}_p75"] = qs[2]

    return stats


def process_structure(file_path):
    """
    Loads an XYZ file and computes all geometric features.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    # 1. Load Geometry
    try:
        # Cite debug_lesson_3: Explicitly specify format='aims' because .xyz extension is misleading
        atoms = ase.io.read(full_path, format="aims")
    except Exception as e:
        # Cite debug_lesson_9: Fail loudly or log error instead of silent failure
        print(f"Error reading {full_path}: {e}")
        return {}

    # 2. Basic Physical Properties
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 0 else 0.0
    n_atoms = len(atoms)

    features = {
        "geo_volume": vol,
        "geo_density": density,
        "geo_num_atoms": n_atoms,
        "geo_vol_per_atom": vol / n_atoms if n_atoms > 0 else 0,
    }

    # 3. Bond Valence Analysis
    bvs_data = calculate_bond_valences(atoms)
    features["bvs_gii"] = bvs_data["gii"]

    scalar_bvs = bvs_data["scalar_bvs"]
    vector_bvs = bvs_data["vector_bvs"]

    # Element-wise Aggregation
    symbols = np.array(atoms.get_chemical_symbols())
    elements = ["Al", "Ga", "In", "O"]

    # 4. Structural Metrics (ECoN, RDF)
    struct_data = calculate_structural_metrics(atoms)
    econ = struct_data["econ"]
    rdf = struct_data["rdf"]

    # Aggregate atom-level metrics by element
    for el in elements:
        mask = symbols == el
        # Scalar BVS
        vals_bvs = scalar_bvs[mask]
        features.update(compute_stats(vals_bvs, prefix=f"bvs_{el}"))

        # Vector BVS
        vals_vec = vector_bvs[mask]
        features.update(compute_stats(vals_vec, prefix=f"vec_bvs_{el}"))

        # ECoN
        vals_econ = econ[mask]
        features.update(compute_stats(vals_econ, prefix=f"econ_{el}"))

    # Global stats for these metrics (across all atoms)
    features.update(compute_stats(scalar_bvs, prefix="bvs_global"))
    features.update(compute_stats(vector_bvs, prefix="vec_bvs_global"))
    features.update(compute_stats(econ, prefix="econ_global"))

    # 5. Weighted Bond Angles
    angle_data = calculate_weighted_bond_angles(atoms)

    # Metal-centered
    m_angles = angle_data["M_centered_angles"]
    m_weights = angle_data["M_centered_weights"]
    features.update(compute_stats(m_angles, weights=m_weights, prefix="angle_M"))

    # Oxygen-centered
    o_angles = angle_data["O_centered_angles"]
    o_weights = angle_data["O_centered_weights"]
    features.update(compute_stats(o_angles, weights=o_weights, prefix="angle_O"))

    # 6. RDF Features
    # Flatten the RDF histograms
    for pair_name, hist in rdf.items():
        # hist is an array of bins
        for i, val in enumerate(hist):
            features[f"rdf_{pair_name}_bin_{i}"] = val

    return features


def generate_features(metadata_df, dataset_key, load_cached_data=True):
    """
    Generates features for the given metadata DataFrame.
    Implements caching.
    """
    cache_file = os.path.join(WORKING_DIR, f"{dataset_key}_features.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Computing features for {dataset_key} ({len(metadata_df)} samples)...")

    feature_list = []

    # Iterate and process
    # Using simple loop for clarity and safety within this environment
    for idx, row in metadata_df.iterrows():
        feats = process_structure(row["file_path"])
        # Add ID for merging safety
        feats["id"] = row["id"]
        feature_list.append(feats)

    # Create DataFrame
    features_df = pd.DataFrame(feature_list)

    # Merge with metadata
    # We assume 'id' is unique and present in both
    # Drop file_path from metadata if not needed, but keep targets if present
    # Actually, we usually return X and y separately or a combined DF.
    # Let's return combined DF.

    # Ensure 'id' is the key
    if "id" in features_df.columns:
        combined_df = pd.merge(metadata_df, features_df, on="id", how="left")
    else:
        # Fallback if something went wrong, though list order is preserved
        print("Warning: ID column missing in features, concatenating by index.")
        combined_df = pd.concat(
            [metadata_df.reset_index(drop=True), features_df.reset_index(drop=True)],
            axis=1,
        )

    # Save to cache
    print(f"Saving features to {cache_file}...")
    combined_df.to_parquet(cache_file, index=False)

    return combined_df
