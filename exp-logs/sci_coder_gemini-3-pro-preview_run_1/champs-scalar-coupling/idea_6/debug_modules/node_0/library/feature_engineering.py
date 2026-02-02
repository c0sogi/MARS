import os
import numpy as np
import pandas as pd
from library import config, data_loader

# =============================================================================
# CONSTANTS
# =============================================================================
ATOM_RADII = config.ATOM_RADII
BOND_TOLERANCE = config.BOND_TOLERANCE


def build_global_adjacency(df_structures):
    """
    Constructs the molecular graph (adjacency list) using adaptive covalent radii.
    Performs a global self-join on structures to identify bonded neighbors.

    Args:
        df_structures (pd.DataFrame): DataFrame with ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z'].

    Returns:
        pd.DataFrame: Directed edge list with columns:
                      ['molecule_name', 'atom_index', 'neighbor_index', 'neighbor_atom',
                       'vec_x', 'vec_y', 'vec_z', 'dist']
    """
    print("  Building global adjacency matrix (vectorized)...")

    # Prepare for self-join
    df_s = df_structures.copy()

    # Global Self-Join to get all potential pairs within molecules
    # This is much faster than iterating molecules
    df_pairs = pd.merge(df_s, df_s, on="molecule_name", suffixes=("_i", "_j"))

    # Remove self-loops
    df_pairs = df_pairs[df_pairs["atom_index_i"] != df_pairs["atom_index_j"]]

    # Calculate Euclidean vectors and distances
    df_pairs["dx"] = df_pairs["x_j"] - df_pairs["x_i"]
    df_pairs["dy"] = df_pairs["y_j"] - df_pairs["y_i"]
    df_pairs["dz"] = df_pairs["z_j"] - df_pairs["z_i"]
    df_pairs["dist"] = np.sqrt(
        df_pairs["dx"] ** 2 + df_pairs["dy"] ** 2 + df_pairs["dz"] ** 2
    )

    # Map Covalent Radii
    df_pairs["rad_i"] = df_pairs["atom_i"].map(ATOM_RADII)
    df_pairs["rad_j"] = df_pairs["atom_j"].map(ATOM_RADII)

    # Apply Adaptive Threshold: Bond if dist < r_i + r_j + tolerance
    mask_bond = df_pairs["dist"] < (
        df_pairs["rad_i"] + df_pairs["rad_j"] + BOND_TOLERANCE
    )
    df_bonds = df_pairs[mask_bond].copy()

    # Select and Rename Columns
    df_bonds = df_bonds.rename(
        columns={
            "atom_index_i": "atom_index",
            "atom_index_j": "neighbor_index",
            "atom_j": "neighbor_atom",
            "dx": "vec_x",
            "dy": "vec_y",
            "dz": "vec_z",
        }
    )

    keep_cols = [
        "molecule_name",
        "atom_index",
        "neighbor_index",
        "neighbor_atom",
        "vec_x",
        "vec_y",
        "vec_z",
        "dist",
    ]

    return df_bonds[keep_cols].reset_index(drop=True)


def compute_node_features(df_bonds):
    """
    Computes Level 1 (Local) and Level 2 (Extended) node features using vectorized operations.

    Args:
        df_bonds (pd.DataFrame): The adjacency list from build_global_adjacency.

    Returns:
        pd.DataFrame: Node features indexed by ['molecule_name', 'atom_index'].
    """
    print("  Computing Level 1 and Level 2 node features...")

    # --- Level 1: Local Neighborhood (Bag of Neighbors) ---
    possible_atoms = ["H", "C", "N", "O", "F"]

    # Create dummy variables for neighbor types
    for atom in possible_atoms:
        df_bonds[f"is_{atom}"] = (df_bonds["neighbor_atom"] == atom).astype(np.int8)

    # Aggregate L1 features
    agg_dict = {f"is_{atom}": "sum" for atom in possible_atoms}
    agg_dict["dist"] = ["mean", "min", "max"]

    # Group by source atom
    df_l1 = df_bonds.groupby(["molecule_name", "atom_index"]).agg(agg_dict)

    # Flatten columns
    df_l1.columns = [
        f"L1_{c[0]}_{c[1]}" if isinstance(c, tuple) else c for c in df_l1.columns
    ]
    df_l1 = df_l1.reset_index()

    # Rename for clarity
    rename_map = {f"L1_is_{atom}_sum": f"L1_count_{atom}" for atom in possible_atoms}
    df_l1 = df_l1.rename(columns=rename_map)

    # --- Level 2: Extended Neighborhood (Vectorized Message Passing) ---
    # Propagate L1 features to neighbors

    # Prepare L1 features to be merged onto 'neighbor_index'
    df_l1_neighbors = df_l1.rename(columns={"atom_index": "neighbor_index"})

    # Rename feature columns to L2 prefix
    feat_cols = [c for c in df_l1.columns if c not in ["molecule_name", "atom_index"]]
    rename_l2 = {c: c.replace("L1_", "L2_neighbor_") for c in feat_cols}
    df_l1_neighbors = df_l1_neighbors.rename(columns=rename_l2)

    # Merge L1 features onto the adjacency list (Message Passing Step)
    df_msg = pd.merge(
        df_bonds[["molecule_name", "atom_index", "neighbor_index"]],
        df_l1_neighbors,
        on=["molecule_name", "neighbor_index"],
        how="left",
    )

    # Aggregate messages at the central atom
    l2_feat_cols = list(rename_l2.values())
    agg_l2 = {}
    for col in l2_feat_cols:
        if "count" in col:
            agg_l2[col] = "sum"  # Sum of counts = total counts in 2-hop
        else:
            agg_l2[col] = "mean"  # Mean of distances

    df_l2 = df_msg.groupby(["molecule_name", "atom_index"]).agg(agg_l2).reset_index()

    # Merge L1 and L2
    df_nodes = pd.merge(df_l1, df_l2, on=["molecule_name", "atom_index"], how="left")

    # Fill NaNs (atoms with no neighbors or no 2-hop neighbors)
    df_nodes = df_nodes.fillna(0)

    return df_nodes


def compute_field_projections(df_pairs, df_bonds, atom_idx_col, prefix):
    """
    Computes Cosine Field Projections: The angle between the coupling axis (atom_0 -> atom_1)
    and the bonds to neighbors for a specific atom.

    Args:
        df_pairs: DataFrame containing ['id', 'molecule_name', atom_idx_col, 'pair_vec_x/y/z']
        df_bonds: Adjacency DataFrame.
        atom_idx_col: 'atom_index_0' or 'atom_index_1'.
        prefix: 'atom_0' or 'atom_1' for column naming.

    Returns:
        pd.DataFrame: Aggregated cosine statistics indexed by 'id'.
    """
    # Merge pairs with bonds to get neighbors of the specific atom
    # This expands the dataframe: one row per pair per neighbor
    df_neighbors = pd.merge(
        df_pairs[
            [
                "id",
                "molecule_name",
                atom_idx_col,
                "pair_vec_x",
                "pair_vec_y",
                "pair_vec_z",
            ]
        ],
        df_bonds[["molecule_name", "atom_index", "vec_x", "vec_y", "vec_z", "dist"]],
        left_on=["molecule_name", atom_idx_col],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )

    # Calculate Dot Product: (Pair Vector) . (Neighbor Bond Vector)
    dot_prod = (
        df_neighbors["pair_vec_x"] * df_neighbors["vec_x"]
        + df_neighbors["pair_vec_y"] * df_neighbors["vec_y"]
        + df_neighbors["pair_vec_z"] * df_neighbors["vec_z"]
    )

    # Calculate Magnitudes
    pair_mag = np.sqrt(
        df_neighbors["pair_vec_x"] ** 2
        + df_neighbors["pair_vec_y"] ** 2
        + df_neighbors["pair_vec_z"] ** 2
    )
    neigh_mag = df_neighbors["dist"]  # Pre-calculated in bonds

    # Calculate Cosine
    # Add epsilon to avoid division by zero
    cosine = dot_prod / (pair_mag * neigh_mag + 1e-9)
    df_neighbors["cosine"] = cosine

    # Aggregate per coupling ID
    agg_stats = (
        df_neighbors.groupby("id")["cosine"]
        .agg(["mean", "min", "max", "std"])
        .fillna(0)
    )
    agg_stats.columns = [f"{prefix}_field_cos_{stat}" for stat in agg_stats.columns]

    return agg_stats


def generate_hierarchical_features(df_metadata, split_name, load_cached_data=True):
    """
    Main pipeline to generate the Vectorized Hierarchical Field-Augmented features.

    Args:
        df_metadata (pd.DataFrame): The train/val/test metadata.
        split_name (str): Name of the split (e.g., 'train', 'test') for caching.
        load_cached_data (bool): Whether to use disk caching.

    Returns:
        pd.DataFrame: The feature-rich DataFrame ready for training/inference.
    """
    # Define cache paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    final_cache_path = os.path.join(cache_dir, f"features_{split_name}.parquet")
    graph_cache_path = os.path.join(cache_dir, "graph_adjacency.parquet")
    node_cache_path = os.path.join(cache_dir, "node_features.parquet")

    # 1. Check Final Cache
    if load_cached_data and os.path.exists(final_cache_path):
        print(f"Loading cached features from {final_cache_path}")
        return pd.read_parquet(final_cache_path)

    print(f"Generating features for {split_name} (VH-FASE Pipeline)...")

    # 2. Load Structures
    df_struct = data_loader.load_structures(load_cached_data=True)

    # 3. Build/Load Global Graph (Adjacency)
    if load_cached_data and os.path.exists(graph_cache_path):
        print("  Loading global graph from cache...")
        df_bonds = pd.read_parquet(graph_cache_path)
    else:
        df_bonds = build_global_adjacency(df_struct)
        if load_cached_data:
            df_bonds.to_parquet(graph_cache_path)

    # 4. Build/Load Node Features (L1 & L2)
    if load_cached_data and os.path.exists(node_cache_path):
        print("  Loading node features from cache...")
        df_nodes = pd.read_parquet(node_cache_path)
    else:
        df_nodes = compute_node_features(df_bonds)
        if load_cached_data:
            df_nodes.to_parquet(node_cache_path)

    # 5. Process Target Pairs (Geometry & Field Projections)
    print("  Processing target pair geometry...")

    # Merge coordinates for Atom 0
    df = pd.merge(
        df_metadata,
        df_struct[["molecule_name", "atom_index", "x", "y", "z"]],
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    df = df.rename(columns={"x": "x0", "y": "y0", "z": "z0"}).drop(
        columns=["atom_index"]
    )

    # Merge coordinates for Atom 1
    df = pd.merge(
        df,
        df_struct[["molecule_name", "atom_index", "x", "y", "z"]],
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    df = df.rename(columns={"x": "x1", "y": "y1", "z": "z1"}).drop(
        columns=["atom_index"]
    )

    # Calculate Pair Vector (Coupling Axis) and Distance
    df["pair_vec_x"] = df["x1"] - df["x0"]
    df["pair_vec_y"] = df["y1"] - df["y0"]
    df["pair_vec_z"] = df["z1"] - df["z0"]
    df["dist"] = np.sqrt(
        df["pair_vec_x"] ** 2 + df["pair_vec_y"] ** 2 + df["pair_vec_z"] ** 2
    )

    # Level 0 Features: Inverse Powers
    df["dist_inv"] = 1.0 / df["dist"]
    df["dist_inv2"] = 1.0 / (df["dist"] ** 2)
    df["dist_inv3"] = 1.0 / (df["dist"] ** 3)

    # Compute Field Projections (Interaction between neighborhood and coupling axis)
    print("  Computing Field Projections...")
    fp_0 = compute_field_projections(df, df_bonds, "atom_index_0", "atom_0")
    fp_1 = compute_field_projections(df, df_bonds, "atom_index_1", "atom_1")

    # Merge Field Projections back to main dataframe
    df = pd.merge(df, fp_0, on="id", how="left")
    df = pd.merge(df, fp_1, on="id", how="left")

    # 6. Merge Node Features
    print("  Merging Hierarchical Node Features...")

    # Merge for Atom 0
    df = pd.merge(
        df,
        df_nodes,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    # Rename columns
    cols_rename_0 = {
        c: f"atom_0_{c}"
        for c in df_nodes.columns
        if c not in ["molecule_name", "atom_index"]
    }
    df = df.rename(columns=cols_rename_0)

    # Merge for Atom 1
    df = pd.merge(
        df,
        df_nodes,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
        suffixes=("", "_dup"),
    )
    cols_rename_1 = {
        c: f"atom_1_{c}"
        for c in df_nodes.columns
        if c not in ["molecule_name", "atom_index"]
    }
    df = df.rename(columns=cols_rename_1)

    # 7. Cleanup and Optimization
    # Drop intermediate geometry columns to save memory
    drop_cols = [
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "pair_vec_x",
        "pair_vec_y",
        "pair_vec_z",
        "atom_index",
        "atom_index_dup",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Downcast types
    df = data_loader._downcast_dtypes(df)

    # Save to cache
    print(f"  Saving processed features to {final_cache_path}")
    df.to_parquet(final_cache_path, index=False)

    return df
