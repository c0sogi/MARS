import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import *
from library.utils import (
    reduce_mem_usage,
    calculate_log_mae,
    timer,
    print_full_precision_metrics,
)


# ==========================================
# Graph Feature Engine
# ==========================================
class GraphFeatureEngine:
    """
    Implements Vectorized Tabular Message Passing to generate deep context node features.
    """

    def __init__(self):
        self.radii = COVALENT_RADII
        self.tolerance = BOND_RADIUS_TOLERANCE

    def _build_bond_graph(self, structures):
        """
        Constructs the bond graph using vectorized operations.
        Performs a global self-join on molecules and filters by covalent radii.

        Args:
            structures (pd.DataFrame): Raw structures data.

        Returns:
            pd.DataFrame: Edge list with distances and vector components.
        """
        # Optimize types for memory
        structures = structures.copy()
        coords = ["x", "y", "z"]
        structures[coords] = structures[coords].astype(np.float32)

        # Global Self-Join on molecule_name to get all potential pairs
        # Note: This creates a large temporary dataframe, but fits within 220GB RAM
        structures_merged = pd.merge(
            structures, structures, on="molecule_name", suffixes=("_0", "_1")
        )

        # Filter Self-loops
        structures_merged = structures_merged[
            structures_merged["atom_index_0"] != structures_merged["atom_index_1"]
        ]

        # Calculate Distances
        d_x = structures_merged["x_0"] - structures_merged["x_1"]
        d_y = structures_merged["y_0"] - structures_merged["y_1"]
        d_z = structures_merged["z_0"] - structures_merged["z_1"]
        structures_merged["dist"] = np.sqrt(d_x**2 + d_y**2 + d_z**2)

        # Determine Connectivity (Adaptive Threshold)
        r0 = structures_merged["atom_0"].map(self.radii).astype(np.float32)
        r1 = structures_merged["atom_1"].map(self.radii).astype(np.float32)

        threshold = r0 + r1 + self.tolerance
        edges = structures_merged[structures_merged["dist"] <= threshold].copy()

        # Add vector components for directional features (relative to atom 0)
        # Vector from 0 to 1 is (x1-x0, ...), which is (-dx, -dy, -dz) based on above calc
        edges["dx"] = -d_x
        edges["dy"] = -d_y
        edges["dz"] = -d_z

        # Clean up
        del structures_merged, d_x, d_y, d_z, r0, r1
        gc.collect()

        return edges

    def _compute_level_1(self, nodes, edges):
        """
        Computes Level 1 Node Features (Local Field).
        Aggregates geometric and topological properties of immediate neighbors.
        """
        # Topological: Bag of Neighbors (One-hot encoding neighbor types)
        for atom in ATOM_TYPES:
            edges[f"is_{atom}"] = (edges["atom_1"] == atom).astype(np.int8)

        # Group by atom_0 (the central atom)
        grp = edges.groupby(["molecule_name", "atom_index_0"])

        # Aggregations
        agg_funcs = {
            "dist": ["mean", "min", "max", "std"],
            "dx": ["mean"],  # Mean bond vector component X (Local Asymmetry)
            "dy": ["mean"],
            "dz": ["mean"],
        }
        for atom in ATOM_TYPES:
            agg_funcs[f"is_{atom}"] = ["sum"]

        features = grp.agg(agg_funcs)

        # Flatten columns
        features.columns = [f"L1_{col[0]}_{col[1]}" for col in features.columns]
        features = features.reset_index()
        features = features.rename(columns={"atom_index_0": "atom_index"})

        # Calculate Valence (Degree)
        features["L1_degree"] = features[[f"L1_is_{a}_sum" for a in ATOM_TYPES]].sum(
            axis=1
        )

        # Merge back to nodes
        nodes = pd.merge(
            nodes, features, on=["molecule_name", "atom_index"], how="left"
        )
        nodes = nodes.fillna(0)

        return nodes

    def _compute_level_2(self, nodes, edges):
        """
        Computes Level 2 Node Features (Message Passing).
        Aggregates the Level 1 features of neighbors to capture extended environment.
        """
        # Prepare neighbor features
        l1_cols = [c for c in nodes.columns if c.startswith("L1_")]
        nodes_l1 = nodes[["molecule_name", "atom_index"] + l1_cols].copy()

        # Merge neighbor features onto edges (join on atom_1)
        edges_enhanced = pd.merge(
            edges[["molecule_name", "atom_index_0", "atom_index_1"]],
            nodes_l1,
            left_on=["molecule_name", "atom_index_1"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )

        # Group by atom_0 and aggregate neighbors' features
        grp = edges_enhanced.groupby(["molecule_name", "atom_index_0"])

        # Compute mean and sum of neighbors' L1 features
        agg_dict = {col: ["mean", "sum"] for col in l1_cols}
        features_l2 = grp.agg(agg_dict)

        # Flatten columns
        features_l2.columns = [f"L2_{col[0]}_{col[1]}" for col in features_l2.columns]
        features_l2 = features_l2.reset_index()
        features_l2 = features_l2.rename(columns={"atom_index_0": "atom_index"})

        # Merge back to nodes
        nodes = pd.merge(
            nodes, features_l2, on=["molecule_name", "atom_index"], how="left"
        )
        nodes = nodes.fillna(0)

        return nodes

    def process_structures(self, load_cached_data=True):
        """
        Orchestrates the feature generation pipeline.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (nodes, edges)
                nodes: pd.DataFrame of atom features.
                edges: pd.DataFrame of bond vectors.
        """
        if (
            load_cached_data
            and os.path.exists(CACHE_STRUCTURES_PROCESSED)
            and os.path.exists(CACHE_EDGES_PROCESSED)
        ):
            print(f"Loading processed structures from {CACHE_STRUCTURES_PROCESSED}...")
            nodes = pd.read_parquet(CACHE_STRUCTURES_PROCESSED)
            print(f"Loading processed edges from {CACHE_EDGES_PROCESSED}...")
            edges = pd.read_parquet(CACHE_EDGES_PROCESSED)
            return nodes, edges

        print("Processing structures from scratch...")

        # Load Raw Structures
        structures = pd.read_csv(STRUCTURES_PATH)

        # Initialize Nodes DataFrame
        nodes = structures[
            ["molecule_name", "atom_index", "atom", "x", "y", "z"]
        ].copy()

        with timer("Graph Construction"):
            edges = self._build_bond_graph(structures)

        with timer("Level 1 Features"):
            nodes = self._compute_level_1(nodes, edges)

        with timer("Level 2 Features"):
            nodes = self._compute_level_2(nodes, edges)

        # Reduce Memory
        nodes = reduce_mem_usage(nodes)
        edges = reduce_mem_usage(edges)

        # Save Cache
        print(f"Saving processed structures to {CACHE_STRUCTURES_PROCESSED}...")
        nodes.to_parquet(CACHE_STRUCTURES_PROCESSED)
        print(f"Saving processed edges to {CACHE_EDGES_PROCESSED}...")
        edges.to_parquet(CACHE_EDGES_PROCESSED)

        return nodes, edges


# ==========================================
# Data Processor
# ==========================================
class DataProcessor:
    def __init__(self, node_features, edges):
        self.node_features = node_features
        self.edges = edges

    def _compute_angular_features(self, df, atom_idx_col, prefix):
        """
        Calculates aggregate angular features (Late Aggregation).
        Computes cosine similarity between the coupling axis and ALL neighbor bonds.
        Cite solution_lesson_node_00028
        """
        # Calculate Coupling Vector C
        if atom_idx_col == "atom_index_0":
            # Vector from 0 to 1
            cx = df["x_1"] - df["x_0"]
            cy = df["y_1"] - df["y_0"]
            cz = df["z_1"] - df["z_0"]
        else:
            # Vector from 1 to 0
            cx = df["x_0"] - df["x_1"]
            cy = df["y_0"] - df["y_1"]
            cz = df["z_0"] - df["z_1"]

        c_norm = np.sqrt(cx**2 + cy**2 + cz**2)

        # Ensure ID for merge
        if "id" not in df.columns:
            df["temp_id"] = np.arange(len(df))
            merge_id = "temp_id"
        else:
            merge_id = "id"

        # Prepare small DF for merge
        df_merge = df[[merge_id, "molecule_name", atom_idx_col]].copy()
        df_merge["cx"] = cx
        df_merge["cy"] = cy
        df_merge["cz"] = cz
        df_merge["c_norm"] = c_norm

        # Merge with edges to get all neighbors
        # Edges columns: molecule_name, atom_index_0 (center), dx, dy, dz (to neighbor)
        merged = pd.merge(
            df_merge,
            self.edges[["molecule_name", "atom_index_0", "dx", "dy", "dz"]],
            left_on=["molecule_name", atom_idx_col],
            right_on=["molecule_name", "atom_index_0"],
            how="left",
            suffixes=("", "_edge"),
        )

        # Calculate Cosine
        # neighbor vector N = (dx, dy, dz)
        dot = (
            merged["cx"] * merged["dx"]
            + merged["cy"] * merged["dy"]
            + merged["cz"] * merged["dz"]
        )
        n_norm = np.sqrt(merged["dx"] ** 2 + merged["dy"] ** 2 + merged["dz"] ** 2)

        merged["cosine"] = dot / (merged["c_norm"] * n_norm + 1e-9)

        # Aggregate
        grp = merged.groupby(merge_id)["cosine"]
        agg = grp.agg(["mean", "min", "max", "std"]).add_prefix(f"{prefix}_cos_")

        # Merge back
        df = pd.merge(df, agg, left_on=merge_id, right_index=True, how="left")

        # Fill NaNs
        fill_cols = [f"{prefix}_cos_{stat}" for stat in ["mean", "min", "max", "std"]]
        df[fill_cols] = df[fill_cols].fillna(0)

        if "temp_id" in df.columns:
            df = df.drop(columns=["temp_id"])

        return df

    def create_dataset(self, metadata_df, is_train=True):
        """
        Merges node features onto coupling pairs and calculates pairwise geometric features.
        """
        # Merge Atom 0 Features
        df = pd.merge(
            metadata_df,
            self.node_features,
            left_on=["molecule_name", "atom_index_0"],
            right_on=["molecule_name", "atom_index"],
            how="left",
            suffixes=("", "_0"),
        ).drop(columns=["atom_index"])

        # Merge Atom 1 Features
        df = pd.merge(
            df,
            self.node_features,
            left_on=["molecule_name", "atom_index_1"],
            right_on=["molecule_name", "atom_index"],
            how="left",
            suffixes=("_0", "_1"),
        ).drop(columns=["atom_index"])

        # Calculate Pairwise Geometry
        dx = df["x_0"] - df["x_1"]
        dy = df["y_0"] - df["y_1"]
        dz = df["z_0"] - df["z_1"]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        df["dist"] = dist
        df["dist_inv"] = 1.0 / dist
        df["dist_inv2"] = 1.0 / (dist**2)
        df["dist_inv3"] = 1.0 / (dist**3)

        # Late Aggregation Angular Features (Cite Lesson 00028)
        df = self._compute_angular_features(df, "atom_index_0", "a0")
        df = self._compute_angular_features(df, "atom_index_1", "a1")

        # Drop non-feature columns
        drop_cols = [
            "id",
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "file_path",
            "x_0",
            "y_0",
            "z_0",
            "x_1",
            "y_1",
            "z_1",
            "atom_0",
            "atom_1",
        ]

        target = None
        if is_train:
            target = df["scalar_coupling_constant"]
            drop_cols.append("scalar_coupling_constant")

        features = df.drop(columns=drop_cols)

        return features, target


# ==========================================
# Main Execution Logic
# ==========================================
def main():
    print("Starting Deep-Context Stratified Gradient Boosting Pipeline...")

    # 1. Feature Engineering
    fe = GraphFeatureEngine()
    node_features, edges = fe.process_structures(load_cached_data=True)
    dp = DataProcessor(node_features, edges)

    # 2. Load Metadata
    train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(VAL_METADATA_PATH)
    test_meta = pd.read_csv(TEST_METADATA_PATH)

    submission_parts = []

    # 3. Stratified Training Loop
    for c_type in COUPLING_TYPES:
        print(f"\n{'='*30}\nProcessing Coupling Type: {c_type}\n{'='*30}")

        # Filter Data
        train_indices = train_meta["type"] == c_type
        val_indices = val_meta["type"] == c_type
        test_indices = test_meta["type"] == c_type

        if not train_indices.any():
            print(f"No training data for {c_type}, skipping.")
            continue

        X_train_meta = train_meta[train_indices]
        X_val_meta = val_meta[val_indices]
        X_test_meta = test_meta[test_indices]

        # Create Datasets
        with timer(f"Dataset Creation ({c_type})"):
            X_train, y_train = dp.create_dataset(X_train_meta, is_train=True)
            X_val, y_val = dp.create_dataset(X_val_meta, is_train=True)
            X_test, _ = dp.create_dataset(X_test_meta, is_train=False)

        # Drop 'type' column if present
        if "type" in X_train.columns:
            X_train = X_train.drop(columns=["type"])
            X_val = X_val.drop(columns=["type"])
            X_test = X_test.drop(columns=["type"])

        print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

        # Train XGBoost
        model = xgb.XGBRegressor(**XGB_PARAMS)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=VERBOSE_EVAL)

        # Predict on Test
        y_pred_test = model.predict(X_test)

        # Collect Submission
        sub_part = pd.DataFrame(
            {"id": X_test_meta["id"], "scalar_coupling_constant": y_pred_test}
        )
        submission_parts.append(sub_part)

        # Save Model
        model.save_model(os.path.join(MODEL_SAVE_DIR, f"{c_type}.json"))

        # Cleanup
        del X_train, y_train, X_val, y_val, X_test, model
        gc.collect()

    # 4. Final Submission
    if submission_parts:
        submission = pd.concat(submission_parts).sort_values("id")
        submission.to_csv(SUBMISSION_FILE_PATH, index=False)
        print(f"\nSubmission saved to {SUBMISSION_FILE_PATH}")
        print(f"Submission Shape: {submission.shape}")
    else:
        print("No predictions generated.")


# Execute Pipeline
if __name__ == "__main__":
    main()
