import os
import sys
import numpy as np
import pandas as pd
import ase.io
from joblib import Parallel, delayed
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    RANDOM_SEED,
)
from library.physics_descriptors import (
    get_pbc_vectors,
    calculate_bond_valences_and_vectors,
    calculate_gii,
    compute_rdf,
    compute_angles,
)


def aggregate_statistics(values: np.ndarray, prefix: str) -> dict:
    """
    Computes percentiles (min, 25%, 50%, 75%, max), mean, and std for a given array.
    """
    stats = {}
    if len(values) == 0:
        for suffix in ["min", "25%", "50%", "75%", "max", "mean", "std"]:
            stats[f"{prefix}_{suffix}"] = 0.0
    else:
        percentiles = np.percentile(values, [0, 25, 50, 75, 100])
        stats[f"{prefix}_min"] = percentiles[0]
        stats[f"{prefix}_25%"] = percentiles[1]
        stats[f"{prefix}_50%"] = percentiles[2]
        stats[f"{prefix}_75%"] = percentiles[3]
        stats[f"{prefix}_max"] = percentiles[4]
        stats[f"{prefix}_mean"] = np.mean(values)
        stats[f"{prefix}_std"] = np.std(values)
    return stats


def process_structure(file_path: str) -> dict:
    """
    Loads a geometry file and extracts physical, chemical, and topological features.
    Uses lower-level functions from physics_descriptors to allow custom aggregation.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    try:
        # Explicitly specify format='aims' to handle FHI-aims files with .xyz extension (Cite debug_lesson_3)
        atoms = ase.io.read(full_path, format="aims")
    except Exception as e:
        # Log error to stderr instead of failing silently (Cite debug_lesson_9)
        print(f"Error reading file {full_path}: {e}", file=sys.stderr)
        return {}

    # 1. Global Physical Descriptors
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    n_atoms = len(atoms)
    density = mass / vol if vol > 0 else 0.0

    # 2. Distances and Vectors (PBC)
    # Using a cutoff of 10.0 to ensure we capture relevant neighbors for BVS and RDF
    rij_vectors, distances = get_pbc_vectors(atoms, cutoff=10.0)

    # 3. Bond Valence Analysis
    scalar_bvs, bvvs_mag = calculate_bond_valences_and_vectors(
        atoms, rij_vectors, distances
    )
    gii = calculate_gii(atoms, scalar_bvs)

    # 4. Aggregation of BVS and BVVS by Element
    symbols = np.array(atoms.get_chemical_symbols())
    chem_stats = {}

    for el in ["Al", "Ga", "In", "O"]:
        mask = symbols == el

        # Scalar BVS stats
        vals_s = scalar_bvs[mask] if np.any(mask) else np.array([])
        chem_stats.update(aggregate_statistics(vals_s, f"bvs_{el}"))

        # Vector BVVS stats
        vals_v = bvvs_mag[mask] if np.any(mask) else np.array([])
        chem_stats.update(aggregate_statistics(vals_v, f"bvvs_{el}"))

    # 5. Radial Distribution Function (RDF)
    rdf_feats = compute_rdf(atoms, distances)

    # 6. Topological Fingerprints (Angles)
    angle_feats = compute_angles(atoms, rij_vectors, distances)

    # Combine all features
    features = {
        "vol_per_atom": vol / n_atoms if n_atoms > 0 else 0,
        "density": density,
        "gii": gii,
        **chem_stats,
        **rdf_feats,
        **angle_feats,
    }

    return features


def build_dataset(
    split: str, load_cached_data: bool = True, n_jobs: int = -1
) -> pd.DataFrame:
    """
    Orchestrates the feature extraction process.
    Loads metadata, processes structures in parallel, merges features, and handles caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from parquet cache.
        n_jobs (int): Number of parallel jobs for extraction.

    Returns:
        pd.DataFrame: The final feature matrix including tabular metadata.
    """
    # Determine paths
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = VAL_METADATA_PATH
    elif split == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    cache_path = os.path.join(WORKING_DIR, f"{split}_features_full.parquet")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {split} set...")

    # Load metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    metadata_df = pd.read_csv(meta_path)

    # Parallel extraction
    # We extract features for each file path
    extracted_features = Parallel(n_jobs=n_jobs)(
        delayed(process_structure)(row["file_path"])
        for _, row in metadata_df.iterrows()
    )

    # Convert to DataFrame
    features_df = pd.DataFrame(extracted_features)

    # Merge with metadata
    # We assume the order is preserved by Parallel (which it is for joblib)
    # Concatenate columns.
    # We keep 'id' and targets from metadata, and tabular features like spacegroup/percent_atom_*.
    # We drop 'file_path' from the final set as it's not a model feature.

    # Identify tabular columns to keep
    # We keep everything from metadata except file_path (and maybe id if we want it as index, but let's keep it as col)
    cols_to_keep = [c for c in metadata_df.columns if c != "file_path"]
    final_df = pd.concat(
        [
            metadata_df[cols_to_keep].reset_index(drop=True),
            features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Handle any NaNs created during extraction (e.g. if a file read failed)
    # Simple strategy: fill with 0 or drop. Given this is a competition, filling with 0 is safer than dropping rows.
    final_df = final_df.fillna(0.0)

    # Save to cache
    print(f"Saving features to {cache_path}")
    final_df.to_parquet(cache_path, index=False)

    return final_df
