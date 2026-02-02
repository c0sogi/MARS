import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from library.config import (
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    WORKING_DIR,
    N_JOBS,
    PERCENTILES,
    RDF_BINS,
    DEBUG,
)
from library.data_loader import load_metadata, read_geometry
from library.descriptors import (
    calculate_macroscopic,
    calculate_rdf,
    calculate_bvs_econ,
    calculate_angles,
)


def compute_percentiles(values, prefix):
    """
    Computes percentiles for a list of values.
    Returns a dictionary with keys {prefix}_p{percentile}.
    """
    res = {}
    if values is None or len(values) == 0:
        for p in PERCENTILES:
            res[f"{prefix}_p{p}"] = np.nan
    else:
        try:
            pcts = np.percentile(values, PERCENTILES)
            for p, val in zip(PERCENTILES, pcts):
                res[f"{prefix}_p{p}"] = val
        except Exception:
            for p in PERCENTILES:
                res[f"{prefix}_p{p}"] = np.nan
    return res


def process_single_structure(row):
    """
    Extracts features for a single crystal structure.
    """
    try:
        # Load Geometry
        atoms = read_geometry(row["file_path"])
        symbols = np.array(atoms.get_chemical_symbols())

        # 1. Macroscopic Features
        macro = calculate_macroscopic(atoms)
        features = {
            "vol_per_atom": macro["volume"] / macro["num_atoms"],
            "density": macro["density"],
            "num_atoms": macro["num_atoms"],
        }

        # Add tabular metadata features
        meta_cols = [
            "spacegroup",
            "percent_atom_al",
            "percent_atom_ga",
            "percent_atom_in",
            "lattice_vector_1_ang",
            "lattice_vector_2_ang",
            "lattice_vector_3_ang",
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
        ]
        for col in meta_cols:
            if col in row:
                features[col] = row[col]

        # 2. Radial Distribution Functions (RDF)
        rdf_data = calculate_rdf(atoms)

        # Metal-Oxygen pairs
        for m in ["Al", "Ga", "In"]:
            key = f"{m}-O"
            hist = rdf_data.get(key, np.zeros(RDF_BINS))
            for i, val in enumerate(hist):
                features[f"RDF_{key}_{i}"] = val

        # Metal-Metal pairs
        metals = ["Al", "Ga", "In"]
        for i, m1 in enumerate(metals):
            for m2 in metals[i:]:
                key = f"{m1}-{m2}"
                hist = rdf_data.get(key, np.zeros(RDF_BINS))
                for bin_idx, val in enumerate(hist):
                    features[f"RDF_{key}_{bin_idx}"] = val

        # 3. Chemical Site Fingerprints (BVS & ECoN)
        bvs_econ = calculate_bvs_econ(atoms)
        bvs_vals = bvs_econ["bvs"]
        econ_vals = bvs_econ["econ"]

        # Group by element and compute percentiles
        for el in ["Al", "Ga", "In", "O"]:
            mask = symbols == el
            if np.any(mask):
                el_bvs = bvs_vals[mask]
                el_econ = econ_vals[mask]
                features.update(compute_percentiles(el_bvs, f"BVS_{el}"))
                features.update(compute_percentiles(el_econ, f"ECoN_{el}"))
            else:
                # If element not present, fill with NaNs
                features.update(compute_percentiles([], f"BVS_{el}"))
                features.update(compute_percentiles([], f"ECoN_{el}"))

        # 4. Topological Fingerprints (Angles)
        angles = calculate_angles(atoms)

        # Intra-polyhedral (O-M-O)
        omo_dict = angles.get("omo", {})
        for m in ["Al", "Ga", "In"]:
            m_angles = omo_dict.get(m, [])
            features.update(compute_percentiles(m_angles, f"Angle_OMO_{m}"))

        # Inter-polyhedral (M-O-M)
        mom_angles = angles.get("mom", [])
        features.update(compute_percentiles(mom_angles, "Angle_MOM"))

        # Add ID for merging/tracking
        features["id"] = row["id"]

        return features

    except Exception as e:
        # In a production pipeline we might log this, but here we just return None to be filtered out
        return None


def process_dataset(
    split: str, load_cached_data: bool = True, debug: bool = DEBUG
) -> pd.DataFrame:
    """
    Orchestrates the feature extraction for a given dataset split.
    Handles caching and parallel processing.
    """
    # Determine cache path
    if split == "train":
        cache_path = TRAIN_FEATURES_PATH
    elif split == "val":
        cache_path = VAL_FEATURES_PATH
    elif split == "test":
        cache_path = TEST_FEATURES_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Load metadata
    print(f"Loading metadata for {split}...")
    meta_df = load_metadata(split, debug=debug)

    # Process in parallel
    print(f"Extracting features for {len(meta_df)} structures (N_JOBS={N_JOBS})...")

    rows = meta_df.to_dict("records")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_structure)(row) for row in rows
    )

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        raise RuntimeError(
            "No features were extracted. Check input data or descriptors."
        )

    # Create DataFrame
    feature_df = pd.DataFrame(results)

    # Ensure ID is integer
    if "id" in feature_df.columns:
        feature_df["id"] = feature_df["id"].astype(int)

    # Sort columns to ensure consistency
    feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)

    # Save to cache
    print(f"Saving features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    feature_df.to_parquet(cache_path, index=False)

    return feature_df
