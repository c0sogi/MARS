import os
import numpy as np
import pandas as pd
import gc
from library import config, utils


def map_atom_info(df, df_structures, suffix):
    """
    Merges structure information (x, y, z, atomic_number) onto the main dataframe
    for a specific atom index column.
    """
    atom_idx_col = f"atom_index_{suffix}"

    # Select relevant columns from structures
    df_struct_subset = df_structures[
        ["molecule_name", "atom_index", "x", "y", "z", "atomic_number"]
    ]

    # Merge
    df = df.merge(
        df_struct_subset,
        left_on=["molecule_name", atom_idx_col],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )

    # Rename columns
    rename_map = {
        "x": f"x_{suffix}",
        "y": f"y_{suffix}",
        "z": f"z_{suffix}",
        "atomic_number": f"atomic_number_{suffix}",
    }
    df = df.rename(columns=rename_map)

    # Drop redundant atom_index column from merge
    df = df.drop(columns=["atom_index"])

    return df


def build_molecular_graph(df_structures):
    """
    Constructs a dataframe representing all chemical bonds in the dataset
    based on covalent radii thresholds.

    Returns:
        pd.DataFrame: Contains columns [molecule_name, atom_index_0, atom_index_1, dist, x_diff, y_diff, z_diff]
                      representing connected atoms.
    """
    print("Constructing molecular graph (bond detection)...")

    # Self-merge structures to get all atom pairs within molecules
    # This is safe because max atoms per molecule is small (~29)
    df_bonds = df_structures.merge(
        df_structures, on="molecule_name", suffixes=("_0", "_1")
    )

    # Remove self-loops
    df_bonds = df_bonds[df_bonds["atom_index_0"] != df_bonds["atom_index_1"]]

    # Calculate distances
    df_bonds["x_diff"] = df_bonds["x_1"] - df_bonds["x_0"]
    df_bonds["y_diff"] = df_bonds["y_1"] - df_bonds["y_0"]
    df_bonds["z_diff"] = df_bonds["z_1"] - df_bonds["z_0"]
    df_bonds["dist"] = np.sqrt(
        df_bonds["x_diff"] ** 2 + df_bonds["y_diff"] ** 2 + df_bonds["z_diff"] ** 2
    )

    # Determine connectivity threshold
    # Threshold = Radius(A0) + Radius(A1) + Tolerance
    # Map radii
    radius_0 = (
        df_bonds["atomic_number_0"].map(config.COVALENT_RADII).fillna(0.7)
    )  # Default to C
    radius_1 = df_bonds["atomic_number_1"].map(config.COVALENT_RADII).fillna(0.7)

    threshold = radius_0 + radius_1 + config.CONNECTIVITY_TOLERANCE

    # Filter bonds
    df_bonds = df_bonds[df_bonds["dist"] <= threshold].copy()

    # Optimize memory
    df_bonds = utils.reduce_mem_usage(df_bonds, verbose=False)

    return df_bonds


def generate_level_2_features(df_bonds):
    """
    Generates node-level features by aggregating neighbor information (Simulated Message Passing).
    """
    print("Generating Level 2 (Node Aggregation) features...")

    # Group by source atom (atom_index_0)
    # We aggregate properties of the neighbors (atom_index_1)

    # Features to aggregate
    # 1. Valence (Count)
    # 2. Mean Bond Distance
    # 3. Mean Neighbor Atomic Number (Chemical Environment)

    aggs = {"atom_index_1": "count", "dist": "mean", "atomic_number_1": "mean"}

    df_node_feats = (
        df_bonds.groupby(["molecule_name", "atom_index_0"]).agg(aggs).reset_index()
    )

    df_node_feats = df_node_feats.rename(
        columns={
            "atom_index_1": "valence",
            "dist": "mean_bond_length",
            "atomic_number_1": "mean_neighbor_atomic_num",
        }
    )

    return df_node_feats


def generate_level_1_features(df_main, df_bonds, suffix):
    """
    Generates geometric features (Cosines) for the neighborhood of a specific atom in the coupling pair.

    Args:
        df_main: The main dataset with coupling pairs.
        df_bonds: The dataframe containing all molecular bonds.
        suffix: '0' or '1', indicating which atom of the coupling pair we are analyzing.
    """
    print(f"Generating Level 1 (Geometric) features for atom_{suffix}...")

    atom_idx_col = f"atom_index_{suffix}"
    other_atom_idx_col = f"atom_index_{1 if suffix=='0' else 0}"

    # We want to find neighbors 'k' of atom 'i' (where i is atom_index_{suffix})
    # such that we can compute the angle j-i-k (where j is the other atom in the coupling)

    # 1. Merge bonds onto the main df to find neighbors of atom i
    # df_main keys: molecule_name, atom_index_{suffix}
    # df_bonds keys: molecule_name, atom_index_0 (source)

    # We only need specific columns from bonds to save memory
    bonds_subset = df_bonds[
        [
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "x_diff",
            "y_diff",
            "z_diff",
            "dist",
        ]
    ]

    # Rename bond columns to avoid collision and clarify meaning
    # bond vector is v_ik (from i to neighbor k)
    bonds_subset = bonds_subset.rename(
        columns={
            "atom_index_0": atom_idx_col,  # Join key
            "atom_index_1": "atom_index_k",  # Neighbor index
            "x_diff": "xk_diff",
            "y_diff": "yk_diff",
            "z_diff": "zk_diff",
            "dist": "dist_ik",
        }
    )

    # Merge: This expands the dataframe! (One row per neighbor k)
    # We use inner join, but keep in mind some atoms might have no other neighbors (terminal H),
    # though usually they are bonded to at least the other atom in the pair if it's a 1J coupling.
    # For 2J/3J, the path is longer.
    df_expanded = df_main.merge(
        bonds_subset, on=["molecule_name", atom_idx_col], how="inner"
    )

    # 2. Filter out the coupling partner itself
    # We want angles with *other* bonds. The angle with the coupling bond itself is 0.
    df_expanded = df_expanded[
        df_expanded["atom_index_k"] != df_expanded[other_atom_idx_col]
    ]

    if df_expanded.empty:
        # Handle case where no auxiliary neighbors exist
        print(f"  No auxiliary neighbors found for atom_{suffix} context.")
        return pd.DataFrame(
            index=df_main.index
        )  # Return empty or handle gracefully later

    # 3. Calculate Cosine
    # Vector v_ij (from i to j) - computed in main df as (xj - xi)
    # But wait, main df has coords. Let's recompute v_ij locally to be safe.
    # The main df has x_0, y_0, z_0 and x_1, y_1, z_1.

    # If suffix is '0', i=0, j=1. v_ij = (x1-x0).
    # If suffix is '1', i=1, j=0. v_ij = (x0-x1).

    if suffix == "0":
        dx_ij = df_expanded["x_1"] - df_expanded["x_0"]
        dy_ij = df_expanded["y_1"] - df_expanded["y_0"]
        dz_ij = df_expanded["z_1"] - df_expanded["z_0"]
    else:
        dx_ij = df_expanded["x_0"] - df_expanded["x_1"]
        dy_ij = df_expanded["y_0"] - df_expanded["y_1"]
        dz_ij = df_expanded["z_0"] - df_expanded["z_1"]

    dist_ij = np.sqrt(dx_ij**2 + dy_ij**2 + dz_ij**2)

    # Dot product: v_ij . v_ik
    dot_prod = (
        dx_ij * df_expanded["xk_diff"]
        + dy_ij * df_expanded["yk_diff"]
        + dz_ij * df_expanded["zk_diff"]
    )

    # Cosine = Dot / (norm_ij * norm_ik)
    cosine = dot_prod / (dist_ij * df_expanded["dist_ik"])

    # Clip for numerical stability
    cosine = cosine.clip(-1.0, 1.0)

    df_expanded["cosine"] = cosine

    # 4. Aggregate back to coupling ID
    # We calculate min, max, mean, std of the cosines
    aggs = df_expanded.groupby("id")["cosine"].agg(["mean", "min", "max", "std"])

    # Rename columns
    aggs.columns = [f"cos_{suffix}_{c}" for c in aggs.columns]

    # Fill NaN std (single neighbor) with 0
    aggs[f"cos_{suffix}_std"] = aggs[f"cos_{suffix}_std"].fillna(0)

    return aggs


def generate_symbolic_features(
    df_metadata, df_structures, split_name, load_cached_data=True
):
    """
    Main pipeline for generating symbolic (tabular) features.

    Args:
        df_metadata (pd.DataFrame): The metadata containing 'id', 'molecule_name', 'atom_index_0/1'.
        df_structures (pd.DataFrame): The structures data.
        split_name (str): 'train', 'val', or 'test' (used for caching).
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: The dataframe with generated features.
    """
    # 1. Caching Logic
    cache_path = os.path.join(
        config.CACHE_DIR, f"symbolic_features_{split_name}.parquet"
    )
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading symbolic features from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Cache load failed ({e}). Regenerating...")

    print(f"Generating symbolic features for {split_name}...")

    # 2. Preprocessing & Level 0 (Distance/Atomic Info)
    # Map atom info for atom 0
    df = map_atom_info(df_metadata, df_structures, "0")
    # Map atom info for atom 1
    df = map_atom_info(df, df_structures, "1")

    # Calculate Distances (Level 0)
    df["dx"] = df["x_0"] - df["x_1"]
    df["dy"] = df["y_0"] - df["y_1"]
    df["dz"] = df["z_0"] - df["z_1"]
    df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)

    # Physics-informed inverse distances
    df["dist_sq"] = df["dist"] ** 2
    df["inv_dist"] = 1.0 / df["dist"]
    df["inv_dist_sq"] = 1.0 / df["dist_sq"]
    df["inv_dist_cu"] = 1.0 / (df["dist"] ** 3)

    # 3. Build Graph & Level 2 (Node Aggregates)
    # We need the bond graph for both Level 1 and Level 2
    df_bonds = build_molecular_graph(df_structures)

    # Generate Node Features (Valence, etc.)
    df_node_feats = generate_level_2_features(df_bonds)

    # Merge Node Features for Atom 0
    df = df.merge(
        df_node_feats,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index_0"],
        how="left",
    ).rename(
        columns={
            "valence": "a0_valence",
            "mean_bond_length": "a0_mean_bond_length",
            "mean_neighbor_atomic_num": "a0_mean_nb_atomic_num",
        }
    )

    # Merge Node Features for Atom 1
    df = df.merge(
        df_node_feats,
        left_on=["molecule_name", "atom_index_1"],
        right_on=[
            "molecule_name",
            "atom_index_0",
        ],  # Join on source index of node feats
        how="left",
    ).rename(
        columns={
            "valence": "a1_valence",
            "mean_bond_length": "a1_mean_bond_length",
            "mean_neighbor_atomic_num": "a1_mean_nb_atomic_num",
        }
    )

    # Fill NaNs for atoms with no bonds (rare but possible in fragments)
    fill_cols = [
        "a0_valence",
        "a1_valence",
        "a0_mean_bond_length",
        "a1_mean_bond_length",
        "a0_mean_nb_atomic_num",
        "a1_mean_nb_atomic_num",
    ]
    df[fill_cols] = df[fill_cols].fillna(0)

    # 4. Level 1 (Geometric/Cosine Features)
    # Calculate for Atom 0
    cos_feats_0 = generate_level_1_features(df, df_bonds, "0")
    df = df.merge(cos_feats_0, on="id", how="left")

    # Calculate for Atom 1
    cos_feats_1 = generate_level_1_features(df, df_bonds, "1")
    df = df.merge(cos_feats_1, on="id", how="left")

    # Fill NaNs for cosine features (occurs if no neighbors)
    cos_cols = [c for c in df.columns if "cos_" in c]
    # For min/max/mean, 0 is a reasonable imputation for orthogonality,
    # but -1 (far away) or 1 (overlap) might bias.
    # Given these are angular features, filling with 0 (orthogonal) is standard neutral.
    df[cos_cols] = df[cos_cols].fillna(0)

    # 5. Cleanup
    # Drop raw coordinate columns to save space, models don't need absolute coords
    drop_cols = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1", "dx", "dy", "dz"]
    # Also drop the molecule_name if not needed for output (but usually kept for tracking)
    # Keeping molecule_name for now as it might be needed for GroupKFold if re-split
    df = df.drop(columns=drop_cols)

    # Reduce memory
    df = utils.reduce_mem_usage(df)

    # 6. Save to Cache
    print(f"Saving symbolic features to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df
