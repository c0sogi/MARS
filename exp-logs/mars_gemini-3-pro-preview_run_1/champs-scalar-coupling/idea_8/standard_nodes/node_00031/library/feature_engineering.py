import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    STRUCTURES_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CACHE_DIR,
    ATOMIC_RADII,
    ATOMIC_NUMBERS,
    RANDOM_STATE,
)
from library.utils import reduce_mem_usage


class FeatureEngineer:
    """
    Implements the Vectorized Multi-Hop Geometric Ensemble feature engineering pipeline.
    """

    def __init__(self, cache_dir=CACHE_DIR, verbose=True):
        self.cache_dir = cache_dir
        self.verbose = verbose
        os.makedirs(self.cache_dir, exist_ok=True)

    def _log(self, message):
        if self.verbose:
            print(f"[FeatureEngineer] {message}")

    def load_structures(self):
        """Loads and processes the structures file."""
        self._log("Loading structures...")
        df = pd.read_csv(STRUCTURES_CSV)

        # Map atomic properties
        df["radius"] = df["atom"].map(ATOMIC_RADII)
        df["atomic_num"] = df["atom"].map(ATOMIC_NUMBERS)

        return reduce_mem_usage(df, verbose=False)

    def build_adjacency(self, structures):
        """
        Constructs the molecular graph using adaptive covalent radii.
        Returns a DataFrame representing edges (atom_i -> atom_j).
        """
        self._log("Building molecular graph (Adaptive Covalent Radii)...")

        # Perform global self-join on molecule_name
        # This is efficient because molecules are small (block-diagonal adjacency)
        atoms = structures[
            [
                "molecule_name",
                "atom_index",
                "atom",
                "x",
                "y",
                "z",
                "radius",
                "atomic_num",
            ]
        ]

        merged = pd.merge(atoms, atoms, on="molecule_name", suffixes=("_0", "_1"))

        # Filter out self-loops
        merged = merged[merged["atom_index_0"] != merged["atom_index_1"]]

        # Calculate Euclidean distance
        p0 = merged[["x_0", "y_0", "z_0"]].values
        p1 = merged[["x_1", "y_1", "z_1"]].values
        merged["dist"] = np.linalg.norm(p0 - p1, axis=1)

        # Apply Adaptive Covalent Radii Threshold
        # Cutoff = r_i + r_j + tolerance (0.5 Angstrom)
        merged["threshold"] = merged["radius_0"] + merged["radius_1"] + 0.5
        adj = merged[merged["dist"] <= merged["threshold"]].copy()

        # Clean up
        adj = adj.drop(columns=["threshold", "radius_0", "radius_1"])

        # Calculate bond vector components (normalized) for later geometry
        adj["dx"] = (adj["x_1"] - adj["x_0"]) / adj["dist"]
        adj["dy"] = (adj["y_1"] - adj["y_0"]) / adj["dist"]
        adj["dz"] = (adj["z_1"] - adj["z_0"]) / adj["dist"]

        self._log(f"Graph constructed. Total edges: {len(adj)}")
        return reduce_mem_usage(adj, verbose=False)

    def compute_node_features(self, structures, adjacency):
        """
        Computes Level 1 (Local) and Level 2 (Message Passing) node features.
        """
        self._log("Computing Node Features (Level 1 & 2)...")

        # --- Level 1: Local Topology ---
        # Group by source atom to get neighbor stats
        grp = adjacency.groupby(["molecule_name", "atom_index_0"])

        # Basic stats
        node_feats = grp.agg(
            {"dist": ["mean", "min", "max", "std"], "atom_index_1": "count"}  # Degree
        )
        node_feats.columns = [
            "_".join(col).strip() for col in node_feats.columns.values
        ]
        node_feats = node_feats.rename(columns={"atom_index_1_count": "degree"})

        # Neighbor type counts (Bag of Neighbors)
        # Pivot the adjacency to count atom types per source atom
        neighbor_types = adjacency.pivot_table(
            index=["molecule_name", "atom_index_0"],
            columns="atom_1",
            aggfunc="size",
            fill_value=0,
        )
        neighbor_types.columns = [f"nb_count_{c}" for c in neighbor_types.columns]

        # Merge Level 1 features
        nodes = pd.merge(
            node_feats, neighbor_types, left_index=True, right_index=True, how="left"
        )

        # --- Level 2: Message Passing (Extended Field) ---
        # We want to aggregate the features of the neighbors onto the central atom.
        # 1. Join Level 1 features back onto the adjacency list (mapping to neighbor 'atom_index_1')
        adj_enriched = pd.merge(
            adjacency[["molecule_name", "atom_index_0", "atom_index_1"]],
            nodes,
            left_on=["molecule_name", "atom_index_1"],
            right_index=True,
            how="left",
        )

        # 2. Group by central atom ('atom_index_0') and aggregate neighbor features
        # We select a subset of important features to aggregate to avoid explosion
        cols_to_agg = ["degree", "dist_mean"] + [
            c for c in nodes.columns if "nb_count" in c
        ]

        l2_agg = adj_enriched.groupby(["molecule_name", "atom_index_0"])[
            cols_to_agg
        ].mean()
        l2_agg.columns = [f"L2_mean_{c}" for c in l2_agg.columns]

        # Final Node Feature Set
        final_nodes = pd.merge(
            nodes, l2_agg, left_index=True, right_index=True, how="left"
        )

        # Reset index to make it mergeable
        final_nodes = final_nodes.reset_index()

        # Add atomic info from structures
        final_nodes = pd.merge(
            final_nodes,
            structures[["molecule_name", "atom_index", "atomic_num", "radius"]],
            left_on=["molecule_name", "atom_index_0"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        ).drop(columns=["atom_index"])

        # Fill NaNs arising from statistical ops (e.g., std of 1 neighbor) or missing neighbors
        final_nodes = final_nodes.fillna(0)

        return reduce_mem_usage(final_nodes, verbose=False)

    def compute_coupling_geometry(self, df, structures, adjacency):
        """
        Computes the 'Late Aggregation' geometric features for specific coupling pairs.
        Calculates angles between the coupling axis and neighbor bonds.
        """
        self._log(f"Computing Late Aggregation Geometry for {len(df)} pairs...")

        # 1. Get coordinates of Atom 0 and Atom 1
        df_geo = pd.merge(
            df,
            structures[["molecule_name", "atom_index", "x", "y", "z"]],
            left_on=["molecule_name", "atom_index_0"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        df_geo = df_geo.rename(columns={"x": "x0", "y": "y0", "z": "z0"}).drop(
            columns=["atom_index"]
        )

        df_geo = pd.merge(
            df_geo,
            structures[["molecule_name", "atom_index", "x", "y", "z"]],
            left_on=["molecule_name", "atom_index_1"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        df_geo = df_geo.rename(columns={"x": "x1", "y": "y1", "z": "z1"}).drop(
            columns=["atom_index"]
        )

        # 2. Compute Coupling Vector (0 -> 1)
        df_geo["dx_c"] = df_geo["x1"] - df_geo["x0"]
        df_geo["dy_c"] = df_geo["y1"] - df_geo["y0"]
        df_geo["dz_c"] = df_geo["z1"] - df_geo["z0"]
        df_geo["dist_c"] = np.sqrt(
            df_geo["dx_c"] ** 2 + df_geo["dy_c"] ** 2 + df_geo["dz_c"] ** 2
        )

        # Normalize coupling vector
        df_geo["nx_c"] = df_geo["dx_c"] / df_geo["dist_c"]
        df_geo["ny_c"] = df_geo["dy_c"] / df_geo["dist_c"]
        df_geo["nz_c"] = df_geo["dz_c"] / df_geo["dist_c"]

        # 3. Define helper for Late Aggregation
        def aggregate_angles(atom_idx_col, suffix):
            # Join couplings with adjacency to get neighbors of the specific atom
            # We filter out the *other* atom in the coupling pair from the neighbors
            # to avoid trivial 1.0 cosine features.

            # Select relevant columns from adjacency
            adj_subset = adjacency[
                ["molecule_name", "atom_index_0", "atom_index_1", "dx", "dy", "dz"]
            ].copy()

            # Merge: For each coupling, get neighbors of atom_X
            merged = pd.merge(
                df_geo[
                    [
                        "id",
                        "molecule_name",
                        atom_idx_col,
                        "atom_index_1" if suffix == "_0" else "atom_index_0",
                        "nx_c",
                        "ny_c",
                        "nz_c",
                    ]
                ],
                adj_subset,
                left_on=["molecule_name", atom_idx_col],
                right_on=["molecule_name", "atom_index_0"],
                how="inner",
                suffixes=("_c", "_n"),
            )

            # Filter: Neighbor cannot be the coupling partner
            other_atom_col = "atom_index_1" if suffix == "_0" else "atom_index_0"
            merged = merged[merged["atom_index_1_n"] != merged[f"{other_atom_col}_c"]]

            # Calculate Cosine Similarity: (Bond_Neighbor . Coupling_Vector)
            # Note: adj['dx'] is normalized. coupling vector is normalized.
            # Dot product is the cosine.
            # For Atom 1, the coupling vector is reversed (1->0) conceptually,
            # or we can just use dot product and take absolute or raw.
            # Let's use raw dot product with the 0->1 vector.
            # If suffix is _1, the vector 0->1 is incoming.
            # Standard practice: Use vector originating from the atom.
            # So for Atom 1, we should use vector 1->0 = - (0->1).

            sign = 1.0 if suffix == "_0" else -1.0

            merged["cos_theta"] = (
                merged["dx"] * (merged["nx_c"] * sign)
                + merged["dy"] * (merged["ny_c"] * sign)
                + merged["dz"] * (merged["nz_c"] * sign)
            )

            # Aggregate
            agg = merged.groupby("id")["cos_theta"].agg(["mean", "max", "min", "std"])
            agg.columns = [f"cos_{c}{suffix}" for c in agg.columns]
            return agg

        # Aggregate for Atom 0
        agg_0 = aggregate_angles("atom_index_0", "_0")

        # Aggregate for Atom 1
        agg_1 = aggregate_angles("atom_index_1", "_1")

        # 4. Merge back to main dataframe
        # Base features
        df_out = df_geo[["id", "dist_c"]].copy()
        df_out["dist_inv2"] = 1.0 / (df_out["dist_c"] ** 2)
        df_out["dist_inv3"] = 1.0 / (df_out["dist_c"] ** 3)

        df_out = df_out.merge(agg_0, on="id", how="left")
        df_out = df_out.merge(agg_1, on="id", how="left")

        # Fill NaNs (atoms with no other neighbors)
        df_out = df_out.fillna(0)

        return reduce_mem_usage(df_out, verbose=False)

    def process_dataset(
        self, metadata_path, structures, nodes, adjacency, is_test=False
    ):
        """
        Orchestrates the feature assembly for a specific split.
        """
        self._log(f"Processing dataset: {metadata_path}")
        df = pd.read_csv(metadata_path)

        # 1. Compute Geometric Interaction Features (Late Aggregation)
        geo_feats = self.compute_coupling_geometry(df, structures, adjacency)

        # 2. Merge Node Features for Atom 0
        df = pd.merge(
            df,
            nodes,
            left_on=["molecule_name", "atom_index_0"],
            right_on=["molecule_name", "atom_index_0"],
            how="left",
        )
        # Rename columns for Atom 0
        rename_map_0 = {
            c: f"{c}_0"
            for c in nodes.columns
            if c not in ["molecule_name", "atom_index_0"]
        }
        df = df.rename(columns=rename_map_0)

        # 3. Merge Node Features for Atom 1
        df = pd.merge(
            df,
            nodes,
            left_on=["molecule_name", "atom_index_1"],
            right_on=["molecule_name", "atom_index_0"],
            how="left",
            suffixes=(None, "_redundant"),
        )

        # Drop redundant column from merge collision
        if "atom_index_0_redundant" in df.columns:
            df = df.drop(columns=["atom_index_0_redundant"])
        # Rename columns for Atom 1
        rename_map_1 = {
            c: f"{c}_1"
            for c in nodes.columns
            if c not in ["molecule_name", "atom_index_0"]
        }
        df = df.rename(columns=rename_map_1)

        # 4. Merge Geometric Features
        df = pd.merge(df, geo_feats, on="id", how="left")

        # 5. Final Cleanup
        # Drop non-feature columns
        drop_cols = ["molecule_name", "atom_index_0", "atom_index_1", "file_path"]
        # Keep 'id', 'type', 'scalar_coupling_constant' (if train)

        if is_test:
            ids = df["id"].values
            X = df.drop(columns=drop_cols + ["id"])
            y = None
            return X, y, ids
        else:
            y = df["scalar_coupling_constant"].values
            X = df.drop(columns=drop_cols + ["id", "scalar_coupling_constant"])
            return X, y, None

    def generate_features(self, load_cached_data=True):
        """
        Main execution method.
        Returns:
            X_train, y_train, X_val, y_val, X_test, ids_test
        """
        # Cache paths
        paths = {
            "X_train": os.path.join(self.cache_dir, "X_train.parquet"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.parquet"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.parquet"),
            "ids_test": os.path.join(self.cache_dir, "ids_test.npy"),
        }

        # Try loading cache
        if load_cached_data:
            if all(os.path.exists(p) for p in paths.values()):
                self._log("Loading features from cache...")
                X_train = pd.read_parquet(paths["X_train"])
                y_train = np.load(paths["y_train"])
                X_val = pd.read_parquet(paths["X_val"])
                y_val = np.load(paths["y_val"])
                X_test = pd.read_parquet(paths["X_test"])
                ids_test = np.load(paths["ids_test"])
                return X_train, y_train, X_val, y_val, X_test, ids_test
            else:
                self._log("Cache missing or incomplete. Computing from scratch...")

        # Compute from scratch
        structures = self.load_structures()
        adjacency = self.build_adjacency(structures)
        nodes = self.compute_node_features(structures, adjacency)

        # Generate splits
        X_train, y_train, _ = self.process_dataset(
            TRAIN_CSV, structures, nodes, adjacency
        )
        X_val, y_val, _ = self.process_dataset(VAL_CSV, structures, nodes, adjacency)
        X_test, _, ids_test = self.process_dataset(
            TEST_CSV, structures, nodes, adjacency, is_test=True
        )

        # Save to cache
        self._log("Saving features to cache...")
        X_train.to_parquet(paths["X_train"])
        np.save(paths["y_train"], y_train)
        X_val.to_parquet(paths["X_val"])
        np.save(paths["y_val"], y_val)
        X_test.to_parquet(paths["X_test"])
        np.save(paths["ids_test"], ids_test)

        return X_train, y_train, X_val, y_val, X_test, ids_test


def get_data(load_cached_data=True):
    """Convenience wrapper for the class."""
    fe = FeatureEngineer()
    return fe.generate_features(load_cached_data=load_cached_data)
