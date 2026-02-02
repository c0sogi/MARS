import os
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    ATOMIC_RADII,
    RANDOM_SEED,
)

# Ensure reproducibility
np.random.seed(RANDOM_SEED)


def load_metadata(split):
    """
    Loads the metadata CSV for a given split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    file_path = os.path.join(METADATA_DIR, f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    return pd.read_csv(file_path)


def load_structures():
    """
    Loads the structures.csv file containing atomic coordinates.

    Returns:
        pd.DataFrame: DataFrame with columns [molecule_name, atom_index, atom, x, y, z].
    """
    file_path = os.path.join(INPUT_DIR, "structures.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structures file not found: {file_path}")
    return pd.read_csv(file_path)


def merge_structures(df, structures):
    """
    Merges atomic coordinates and types into the main dataframe for both atom_0 and atom_1.

    Args:
        df (pd.DataFrame): Main dataframe containing coupling pairs.
        structures (pd.DataFrame): Structures dataframe.

    Returns:
        pd.DataFrame: Dataframe with added columns x0, y0, z0, atom_0, x1, y1, z1, atom_1.
    """

    # Helper to merge for a specific atom index
    def _merge_atom(base_df, struct_df, atom_idx_col, suffix):
        merged = pd.merge(
            base_df,
            struct_df,
            left_on=["molecule_name", atom_idx_col],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        rename_map = {
            "x": f"x{suffix}",
            "y": f"y{suffix}",
            "z": f"z{suffix}",
            "atom": f"atom_{suffix}",
        }
        merged = merged.rename(columns=rename_map)
        return merged.drop(columns=["atom_index"])

    # Merge for atom_0
    df = _merge_atom(df, structures, "atom_index_0", "0")
    # Merge for atom_1
    df = _merge_atom(df, structures, "atom_index_1", "1")

    return df


def _compute_bond_graph(structures):
    """
    Internal helper to compute a bond graph (adjacency list) for all molecules
    based on atomic radii and distance thresholds.
    """
    # Self-join structures on molecule_name to get all potential pairs
    # We select only necessary columns to save memory
    s_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
    s_left = structures[s_cols]
    s_right = structures[s_cols]

    # Merge to create pairwise combinations within molecules
    bonds = pd.merge(s_left, s_right, on="molecule_name", suffixes=("", "_neighbor"))

    # Filter out self-loops
    bonds = bonds[bonds["atom_index"] != bonds["atom_index_neighbor"]]

    # Calculate Euclidean distances
    bonds["dx"] = bonds["x_neighbor"] - bonds["x"]
    bonds["dy"] = bonds["y_neighbor"] - bonds["y"]
    bonds["dz"] = bonds["z_neighbor"] - bonds["z"]
    bonds["dist"] = np.sqrt(bonds["dx"] ** 2 + bonds["dy"] ** 2 + bonds["dz"] ** 2)

    # Determine bond threshold based on atomic radii
    # Threshold = r_i + r_j + 0.3 Angstrom (tolerance for covalent bonds)
    bonds["rad"] = bonds["atom"].map(ATOMIC_RADII)
    bonds["rad_neighbor"] = bonds["atom_neighbor"].map(ATOMIC_RADII)
    bonds["threshold"] = bonds["rad"] + bonds["rad_neighbor"] + 0.3

    # Filter for bonded pairs
    bonds = bonds[bonds["dist"] < bonds["threshold"]].copy()

    # Return relevant columns for feature engineering
    return bonds[
        [
            "molecule_name",
            "atom_index",
            "atom_index_neighbor",
            "atom_neighbor",
            "dx",
            "dy",
            "dz",
            "dist",
        ]
    ]


def _compute_angular_features(df, bonds, atom_idx_col, vec_ref_prefix):
    """
    Computes cosine similarity statistics (mean, min, max) between the coupling axis
    and the bond vectors of neighbors for a specific atom.
    """
    # Prepare keys and reference vectors from main dataframe
    # We need 'id' to aggregate back later
    cols = [
        "id",
        "molecule_name",
        atom_idx_col,
        f"{vec_ref_prefix}_x",
        f"{vec_ref_prefix}_y",
        f"{vec_ref_prefix}_z",
    ]
    df_keys = df[cols].copy()

    # Join with bond graph to find neighbors of the target atom
    # Inner join: only consider atoms that actually have bonded neighbors
    merged = pd.merge(
        df_keys,
        bonds,
        left_on=["molecule_name", atom_idx_col],
        right_on=["molecule_name", "atom_index"],
        how="inner",
    )

    # Calculate Cosine Similarity
    # Vector A: Reference Vector (Coupling Axis)
    # Vector B: Bond Vector (Atom -> Neighbor)
    # Cosine = (A . B) / (|A| * |B|)

    dot_prod = (
        merged[f"{vec_ref_prefix}_x"] * merged["dx"]
        + merged[f"{vec_ref_prefix}_y"] * merged["dy"]
        + merged[f"{vec_ref_prefix}_z"] * merged["dz"]
    )

    mag_ref = np.sqrt(
        merged[f"{vec_ref_prefix}_x"] ** 2
        + merged[f"{vec_ref_prefix}_y"] ** 2
        + merged[f"{vec_ref_prefix}_z"] ** 2
    )
    mag_neighbor = merged["dist"]

    # Avoid division by zero (though unlikely given bond constraints)
    cosine = dot_prod / (mag_ref * mag_neighbor + 1e-9)
    cosine = cosine.clip(-1.0, 1.0)

    merged["cosine"] = cosine

    # Aggregate statistics per coupling pair (id)
    suffix = "_0" if "0" in atom_idx_col else "_1"
    agg_funcs = {"cosine": ["mean", "min", "max"]}
    features = merged.groupby("id").agg(agg_funcs)

    # Flatten column names
    features.columns = [f"cos_{stat}{suffix}" for stat in features.columns.droplevel(0)]

    return features


def _compute_topological_features(df, bonds, atom_idx_col):
    """
    Computes 'Bag of Neighbors' features: counts of each atom type bonded to the target atom,
    plus distance statistics (min, mean) to neighbors.
    """
    # Get neighbors for the target atoms
    # We include 'dist' to calculate distance statistics (Cite solution_lesson_node_00003)
    relevant_bonds = pd.merge(
        df[["id", "molecule_name", atom_idx_col]],
        bonds[["molecule_name", "atom_index", "atom_neighbor", "dist"]],
        left_on=["molecule_name", atom_idx_col],
        right_on=["molecule_name", "atom_index"],
        how="inner",
    )

    # 1. Counts (Pivot table to count occurrences of each atom type per ID)
    counts = (
        relevant_bonds.groupby(["id", "atom_neighbor"]).size().unstack(fill_value=0)
    )

    # Ensure all expected atom types are present as columns
    possible_atoms = ["C", "H", "N", "O", "F"]
    for atom in possible_atoms:
        if atom not in counts.columns:
            counts[atom] = 0

    # Rename count columns
    suffix = "_0" if "0" in atom_idx_col else "_1"
    counts.columns = [f"n_{atom}{suffix}" for atom in counts.columns]

    # Add degree (total number of neighbors)
    counts[f"degree{suffix}"] = counts.sum(axis=1)

    # 2. Distance Statistics (Min and Mean distance to neighbors)
    dist_stats = relevant_bonds.groupby("id")["dist"].agg(["min", "mean"])
    dist_stats.columns = [f"min_dist_neigh{suffix}", f"mean_dist_neigh{suffix}"]

    # Merge stats into counts
    features = counts.join(dist_stats, how="left")

    return features


def process_data(split, load_cached_data=True):
    """
    Main processing function to generate the feature-rich dataset.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Processed dataframe ready for training/inference.
    """
    # 1. Cache Mechanism
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, f"{split}_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data from scratch...")

    # 2. Load Raw Data
    df = load_metadata(split)
    structures = load_structures()

    # 3. Merge Coordinates
    df = merge_structures(df, structures)

    # 4. Basic Distance Features
    df["dx"] = df["x0"] - df["x1"]
    df["dy"] = df["y0"] - df["y1"]
    df["dz"] = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)

    # Inverse distance features for field decay modeling
    df["dist_inv"] = 1.0 / df["dist"]
    df["dist_inv2"] = 1.0 / (df["dist"] ** 2)
    df["dist_inv3"] = 1.0 / (df["dist"] ** 3)

    # 5. Compute Bond Graph (Full Molecule Context)
    # This is required for topological and angular features
    bonds = _compute_bond_graph(structures)

    # 6. Angular Features (Cosine Projections)
    # Define reference vectors (Coupling Axis)
    # Ref for Atom 0: Vector pointing from 0 to 1
    df["v_ref_x"] = -df["dx"]  # (x1 - x0)
    df["v_ref_y"] = -df["dy"]
    df["v_ref_z"] = -df["dz"]

    # Ref for Atom 1: Vector pointing from 1 to 0
    df["v_ref_neg_x"] = df["dx"]  # (x0 - x1)
    df["v_ref_neg_y"] = df["dy"]
    df["v_ref_neg_z"] = df["dz"]

    # Compute features
    ang_feat_0 = _compute_angular_features(df, bonds, "atom_index_0", "v_ref")
    ang_feat_1 = _compute_angular_features(df, bonds, "atom_index_1", "v_ref_neg")

    # Merge features
    df = df.merge(ang_feat_0, on="id", how="left")
    df = df.merge(ang_feat_1, on="id", how="left")

    # Fill NaNs for atoms with no neighbors (isolated or terminal without other bonds)
    ang_cols = [c for c in df.columns if c.startswith("cos_")]
    df[ang_cols] = df[ang_cols].fillna(0.0)

    # 7. Topological Features (Bag of Neighbors)
    topo_feat_0 = _compute_topological_features(df, bonds, "atom_index_0")
    topo_feat_1 = _compute_topological_features(df, bonds, "atom_index_1")

    df = df.merge(topo_feat_0, on="id", how="left")
    df = df.merge(topo_feat_1, on="id", how="left")

    # Fill NaNs for counts (0 neighbors)
    topo_cols = [c for c in df.columns if c.startswith("n_") or c.startswith("degree")]
    df[topo_cols] = df[topo_cols].fillna(0)

    # 8. Cleanup
    # Drop intermediate vector columns
    drop_cols = [
        "v_ref_x",
        "v_ref_y",
        "v_ref_z",
        "v_ref_neg_x",
        "v_ref_neg_y",
        "v_ref_neg_z",
        "dx",
        "dy",
        "dz",
    ]
    df = df.drop(columns=drop_cols)

    # 9. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    df.to_parquet(cache_path)

    return df
