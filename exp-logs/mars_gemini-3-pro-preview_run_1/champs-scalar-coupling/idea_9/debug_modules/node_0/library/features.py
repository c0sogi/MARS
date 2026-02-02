import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    DATA_PATHS,
    COVALENT_RADII,
    BOND_RADIUS_TOLERANCE,
    ATOM_TYPES,
    WORKING_DIR,
    RANDOM_STATE,
)
from library.utils import reduce_mem_usage


class GraphEngine:
    """
    Engine for generating Vectorized Multi-Hop Geometric features.
    Handles graph construction, message passing simulation, and geometric feature extraction.
    """

    def __init__(self, structures_path=DATA_PATHS["structures"]):
        self.structures_path = structures_path
        self.structures = None
        self.edges = None
        self.node_features = None

    def load_and_process_structures(self):
        """
        Loads structures and builds the molecular graph using adaptive covalent radii.
        """
        print("Loading structures...")
        df = pd.read_csv(self.structures_path)
        df = reduce_mem_usage(df, verbose=False)
        self.structures = df

        # 1. Build Adjacency List (Global Self-Join)
        print("Building molecular graph (Adaptive Covalent Radii)...")
        # Rename for merge
        atoms = df.rename(
            columns={"atom_index": "idx", "atom": "type", "x": "x", "y": "y", "z": "z"}
        )

        # Self join on molecule to get all potential pairs
        # We only keep pairs where idx_i != idx_j.
        # Note: This produces a directed graph representation (i->j and j->i both appear)
        merged = atoms.merge(atoms, on="molecule_name", suffixes=("_i", "_j"))
        merged = merged[merged["idx_i"] != merged["idx_j"]]

        # Calculate Distances
        dx = merged["x_i"] - merged["x_j"]
        dy = merged["y_i"] - merged["y_j"]
        dz = merged["z_i"] - merged["z_j"]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Adaptive Thresholding
        # Get radii
        r_i = merged["type_i"].map(COVALENT_RADII).astype(np.float32)
        r_j = merged["type_j"].map(COVALENT_RADII).astype(np.float32)

        # Filter edges
        mask = dist < (r_i + r_j + BOND_RADIUS_TOLERANCE)
        self.edges = merged[mask].copy()
        self.edges["dist"] = dist[mask]

        # Save vector components for angle calculations later
        self.edges["vec_x"] = dx[mask]
        self.edges["vec_y"] = dy[mask]
        self.edges["vec_z"] = dz[mask]

        # Clean up
        del merged, dx, dy, dz, r_i, r_j, mask
        gc.collect()

        print(f"Graph constructed. Total edges: {len(self.edges)}")
        self.edges = reduce_mem_usage(self.edges, verbose=False)

    def compute_node_features(self):
        """
        Computes Level 1 (Topological) and Level 2 (Message Passing) node features.
        """
        print("Computing Node Features (Level 1 & 2)...")

        # --- Level 1: Local Topology (Bag of Neighbors) ---
        # One-hot encode neighbor types in the edge list
        # We group by source node (molecule_name, idx_i)

        # Create dummy columns for pivot
        for atom_type in ATOM_TYPES:
            self.edges[f"is_{atom_type}"] = (self.edges["type_j"] == atom_type).astype(
                np.int8
            )

        # Aggregations on edges to get node properties
        # Count neighbors of each type, mean distance to neighbors
        aggs = {f"is_{t}": "sum" for t in ATOM_TYPES}
        aggs["dist"] = ["mean", "min", "std"]

        # Group by central atom (i)
        node_feats = self.edges.groupby(["molecule_name", "idx_i"]).agg(aggs)
        node_feats.columns = [f"L1_{c[0]}_{c[1]}" for c in node_feats.columns]
        node_feats = node_feats.reset_index()

        # Add central atom type info
        structures_lite = self.structures[
            ["molecule_name", "atom_index", "atom"]
        ].rename(columns={"atom_index": "idx_i", "atom": "center_type"})
        node_feats = node_feats.merge(
            structures_lite, on=["molecule_name", "idx_i"], how="right"
        )

        # Fill NaNs for atoms with no bonds (rare but possible in fragments/ions)
        node_feats = node_feats.fillna(0)

        # One-hot encode center type
        for atom_type in ATOM_TYPES:
            node_feats[f"center_is_{atom_type}"] = (
                node_feats["center_type"] == atom_type
            ).astype(np.int8)
        node_feats.drop(columns=["center_type"], inplace=True)

        # --- Level 2: Message Passing (Neighbors' Neighbors) ---
        # We want to aggregate the L1 features of neighbors (j) back to i.
        # Join L1 features to edges on j

        # Prepare L1 features for join (rename to _j)
        l1_cols = [c for c in node_feats.columns if c not in ["molecule_name", "idx_i"]]
        node_feats_j = node_feats.rename(columns={"idx_i": "idx_j"})

        # Merge edges with node features of the neighbor (j)
        # We only need the mapping (mol, i) -> (mol, j)
        edge_map = self.edges[["molecule_name", "idx_i", "idx_j"]]

        # Merge
        msg_pass = edge_map.merge(
            node_feats_j, on=["molecule_name", "idx_j"], how="left"
        )

        # Group by i and aggregate
        # We take the mean of neighbors' features
        l2_aggs = {c: "mean" for c in l1_cols}
        l2_feats = msg_pass.groupby(["molecule_name", "idx_i"]).agg(l2_aggs)
        l2_feats.columns = [f"L2_{c}" for c in l2_feats.columns]
        l2_feats = l2_feats.reset_index()

        # Final Node Features: Merge L1 and L2
        self.node_features = node_feats.merge(
            l2_feats, on=["molecule_name", "idx_i"], how="left"
        )
        self.node_features = self.node_features.fillna(0)

        print(f"Node features computed. Shape: {self.node_features.shape}")
        self.node_features = reduce_mem_usage(self.node_features, verbose=False)

        # Cleanup
        del node_feats, node_feats_j, msg_pass, l2_feats
        gc.collect()

    def compute_pair_features(self, pairs_df):
        """
        Computes geometric features for specific atom pairs.
        Includes Level 0 (Distance) and Level 1 Geometry (Angular Aggregations).
        """
        print("Computing Pairwise Geometric Features...")

        # 1. Coordinate Merge & Distance (Level 0)
        # Map coordinates for atom 0 and atom 1
        struct_map = self.structures[["molecule_name", "atom_index", "x", "y", "z"]]

        df = pairs_df.merge(
            struct_map.rename(
                columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
            ),
            on=["molecule_name", "atom_index_0"],
            how="left",
        )
        df = df.merge(
            struct_map.rename(
                columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
            ),
            on=["molecule_name", "atom_index_1"],
            how="left",
        )

        # Calculate Coupling Axis Vector (u = p1 - p0)
        df["u_x"] = df["x1"] - df["x0"]
        df["u_y"] = df["y1"] - df["y0"]
        df["u_z"] = df["z1"] - df["z0"]
        df["dist"] = np.sqrt(df["u_x"] ** 2 + df["u_y"] ** 2 + df["u_z"] ** 2)

        # Inverse distance powers
        df["dist_inv"] = 1.0 / df["dist"]
        df["dist_inv2"] = 1.0 / (df["dist"] ** 2)
        df["dist_inv3"] = 1.0 / (df["dist"] ** 3)

        # Normalize coupling axis vector
        df["u_x"] = df["u_x"] / df["dist"]
        df["u_y"] = df["u_y"] / df["dist"]
        df["u_z"] = df["u_z"] / df["dist"]

        # 2. Level 1 Geometry: Angular Aggregation (Late Aggregation)
        # We need to compute angles between coupling axis and neighbors of atom 0, and neighbors of atom 1.

        # --- Process Atom 0 Neighbors ---
        # Get neighbors of atom 0 from edges
        # We join edges on (molecule_name, atom_index_0) -> (molecule_name, idx_i)
        # The edge vector v is (neighbor - atom0).
        # In self.edges, if idx_i is atom0, then vec_x is (x_i - x_j) = (atom0 - neighbor).
        # Wait, in build_graph: dx = x_i - x_j. So vec is pointing FROM neighbor TO atom0 (if we view j as neighbor).
        # Actually: dx = merged["x_i"] - merged["x_j"]. i is source, j is target. Vector is i -> j? No, it's position difference.
        # Let's be precise.
        # We want vector FROM atom0 TO neighbor.
        # In edges: idx_i is atom0. idx_j is neighbor.
        # dx stored is x_i - x_j. This is Vector(j->i).
        # We want Vector(i->j) = -Vector(j->i).
        # So v_x = -edges["vec_x"], etc.

        # Subset edges to relevant molecules to save memory? No, global join is safer/easier if memory allows.
        # But pairs_df is split (train/test). We should process only relevant pairs.
        # Strategy: We can't easily loop. We must join.

        # To avoid massive explosion, we do this in two passes (Atom 0, Atom 1)

        def aggregate_angles(atom_idx_col, vector_sign_wrt_coupling):
            """
            atom_idx_col: 'atom_index_0' or 'atom_index_1'
            vector_sign_wrt_coupling: 1.0 if we want angle with u, -1.0 if we want angle with -u (for atom 1)
            """
            # Select relevant edges: source is the atom in the pair
            # We merge pairs_df[['id', 'molecule_name', atom_idx_col]] with edges on idx_i
            # This expands the dataframe to (Num_Pairs * Avg_Degree) rows.

            # Optimization: Filter edges to only those molecules in pairs_df
            relevant_mols = pairs_df["molecule_name"].unique()
            relevant_edges = self.edges[self.edges["molecule_name"].isin(relevant_mols)]

            # Prepare Pair Info for merge
            # We need the coupling axis vector (u) to compute dot product
            pair_cols = ["id", "molecule_name", atom_idx_col, "u_x", "u_y", "u_z"]
            pairs_mini = df[pair_cols]

            # Merge
            # Left on: [mol, atom_idx], Right on: [mol, idx_i]
            merged = pairs_mini.merge(
                relevant_edges,
                left_on=["molecule_name", atom_idx_col],
                right_on=["molecule_name", "idx_i"],
                how="inner",  # Only consider atoms that have neighbors
            )

            # Vector to neighbor (i -> j)
            # stored vec is i - j. So i->j is -(i-j) = j - i.
            # So v_x = -merged["vec_x"]
            v_x = -merged["vec_x"]
            v_y = -merged["vec_y"]
            v_z = -merged["vec_z"]
            v_dist = merged["dist"]  # length

            # Normalize v
            v_x = v_x / v_dist
            v_y = v_y / v_dist
            v_z = v_z / v_dist

            # Coupling axis u is already normalized in pairs_mini
            # Calculate Cosine
            # cos = u . v
            # Apply sign: For atom 0, we use u. For atom 1, we use -u (vector from 1 to 0).
            # Actually, standard convention is usually looking "outward" from the bond.
            # Let's stick to the prompt: "cosine similarity between the coupling axis and neighbor bond".
            # Coupling axis is usually p0 -> p1.
            # For atom 0: angle between (p0->p1) and (p0->neighbor).
            # For atom 1: angle between (p1->p0) and (p1->neighbor).
            # So for atom 1, we use -u.

            dot = (
                merged["u_x"] * v_x + merged["u_y"] * v_y + merged["u_z"] * v_z
            ) * vector_sign_wrt_coupling

            # Clip for numerical stability
            dot = dot.clip(-1.0, 1.0)

            merged["cos_angle"] = dot

            # Aggregate
            aggs = (
                merged.groupby("id")["cos_angle"]
                .agg(["mean", "min", "max", "std"])
                .fillna(0)
            )
            suffix = "_0" if "0" in atom_idx_col else "_1"
            aggs.columns = [f"cos_angle{suffix}_{c}" for c in aggs.columns]

            return aggs

        # Aggregations for Atom 0 (u vs p0->n)
        aggs_0 = aggregate_angles("atom_index_0", 1.0)
        df = df.merge(aggs_0, on="id", how="left")

        # Aggregations for Atom 1 (-u vs p1->n)
        aggs_1 = aggregate_angles("atom_index_1", -1.0)
        df = df.merge(aggs_1, on="id", how="left")

        # Fill NaNs (atoms with no neighbors or fragments)
        angle_cols = [c for c in df.columns if "cos_angle" in c]
        df[angle_cols] = df[angle_cols].fillna(0)

        # 3. Merge Node Features (Level 1 & 2)
        # Merge for Atom 0
        df = df.merge(
            self.node_features.rename(columns={"idx_i": "atom_index_0"}),
            on=["molecule_name", "atom_index_0"],
            how="left",
        )
        # Rename columns for Atom 0
        node_cols = [
            c for c in self.node_features.columns if c not in ["molecule_name", "idx_i"]
        ]
        rename_map_0 = {c: f"a0_{c}" for c in node_cols}
        df.rename(columns=rename_map_0, inplace=True)

        # Merge for Atom 1
        df = df.merge(
            self.node_features.rename(columns={"idx_i": "atom_index_1"}),
            on=["molecule_name", "atom_index_1"],
            how="left",
        )
        # Rename columns for Atom 1
        rename_map_1 = {c: f"a1_{c}" for c in node_cols}
        df.rename(columns=rename_map_1, inplace=True)

        # Drop coordinate columns to save space
        drop_cols = ["x0", "y0", "z0", "x1", "y1", "z1", "u_x", "u_y", "u_z"]
        df.drop(columns=drop_cols, inplace=True)

        return reduce_mem_usage(df, verbose=False)


def generate_features(load_cached_data=True):
    """
    Main function to generate features for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df) with generated features.
    """
    # Check cache
    cache_exists = (
        os.path.exists(DATA_PATHS["train_features"])
        and os.path.exists(DATA_PATHS["val_features"])
        and os.path.exists(DATA_PATHS["test_features"])
    )

    if load_cached_data and cache_exists:
        print("Loading features from cache...")
        train_df = pd.read_parquet(DATA_PATHS["train_features"])
        val_df = pd.read_parquet(DATA_PATHS["val_features"])
        test_df = pd.read_parquet(DATA_PATHS["test_features"])
        return train_df, val_df, test_df

    print("Generating features from scratch...")

    # Initialize Engine
    engine = GraphEngine()

    # 1. Load and Process Structures (Global)
    # Check if intermediate structures/edges are cached
    if (
        load_cached_data
        and os.path.exists(DATA_PATHS["graph_edges"])
        and os.path.exists(DATA_PATHS["node_features"])
    ):
        print("Loading graph cache...")
        engine.structures = pd.read_parquet(DATA_PATHS["structures_processed"])
        engine.edges = pd.read_parquet(DATA_PATHS["graph_edges"])
        engine.node_features = pd.read_parquet(DATA_PATHS["node_features"])
    else:
        engine.load_and_process_structures()
        engine.compute_node_features()

        # Save graph cache
        print("Saving graph cache...")
        engine.structures.to_parquet(DATA_PATHS["structures_processed"])
        engine.edges.to_parquet(DATA_PATHS["graph_edges"])
        engine.node_features.to_parquet(DATA_PATHS["node_features"])

    # 2. Process Metadata Splits
    def process_split(meta_path, save_path):
        print(f"Processing split: {meta_path}")
        df = pd.read_csv(meta_path)
        df = reduce_mem_usage(df, verbose=False)

        # Compute features
        df_features = engine.compute_pair_features(df)

        # Save
        print(f"Saving to {save_path}...")
        df_features.to_parquet(save_path)
        return df_features

    train_df = process_split(DATA_PATHS["train_meta"], DATA_PATHS["train_features"])
    val_df = process_split(DATA_PATHS["val_meta"], DATA_PATHS["val_features"])
    test_df = process_split(DATA_PATHS["test_meta"], DATA_PATHS["test_features"])

    print("Feature generation complete.")
    return train_df, val_df, test_df
