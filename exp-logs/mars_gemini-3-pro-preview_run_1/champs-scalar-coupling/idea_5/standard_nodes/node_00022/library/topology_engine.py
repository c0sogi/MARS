import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from library import config, utils

# Disable chained assignment warning
pd.options.mode.chained_assignment = None


class TopologyEngine:
    """
    Vectorized Engine for Graph Feature Engineering.
    Replaces iterative processing with global DataFrame operations to maximize data throughput.
    Cite solution_lesson_node_00021: Prioritize Vectorized Feature Engineering.
    """

    def __init__(self, structures_df, verbose=True):
        self.structures_df = structures_df
        self.verbose = verbose

        # Pre-calculate atom properties
        self.structures_df["en"] = self.structures_df["atom"].map(
            config.ATOM_ELECTRONEGATIVITY
        )
        self.structures_df["rad"] = self.structures_df["atom"].map(
            config.COVALENT_RADII
        )

        # Pre-compute neighbor features globally (Cite solution_lesson_node_00004)
        if self.verbose:
            print("Pre-computing neighbor features...")
        self.neighbor_counts = self._compute_neighbor_features()

    def _compute_neighbor_features(self):
        """
        Computes Bag of Neighbors features using vectorized self-joins.
        Cite solution_lesson_node_00004: Simulating Graph Message Passing with Tabular Self-Joins.
        """
        # Prepare atoms DataFrame
        atoms = self.structures_df[
            ["molecule_name", "atom_index", "x", "y", "z", "atom", "rad"]
        ].copy()

        # Self-join to create all potential pairs within molecules
        # This effectively creates an edge list
        bonds = pd.merge(atoms, atoms, on="molecule_name", suffixes=("", "_neigh"))

        # Filter out self-loops
        bonds = bonds[bonds["atom_index"] != bonds["atom_index_neigh"]]

        # Calculate Euclidean distance
        bonds["dist"] = np.sqrt(
            (bonds["x"] - bonds["x_neigh"]) ** 2
            + (bonds["y"] - bonds["y_neigh"]) ** 2
            + (bonds["z"] - bonds["z_neigh"]) ** 2
        )

        # Apply physical bond threshold (Cite solution_lesson_node_00020: Adaptive Thresholds)
        # Condition: dist < r_i + r_j + tolerance
        threshold = bonds["rad"] + bonds["rad_neigh"] + config.BOND_TOLERANCE
        bonds = bonds[bonds["dist"] <= threshold]

        # Aggregate: Count neighbors by type for each atom
        # Group by [molecule, atom] and pivot on neighbor type
        counts = pd.crosstab(
            [bonds["molecule_name"], bonds["atom_index"]], bonds["atom_neigh"]
        )

        # Rename columns to indicate Level 1 features
        counts.columns = [f"L1_{c}" for c in counts.columns]

        # Ensure all atomic types are present (even if count is 0 globally)
        for atom_type in config.ATOMIC_NUMBERS.keys():
            col = f"L1_{atom_type}"
            if col not in counts.columns:
                counts[col] = 0

        return counts

    def generate_features(self, metadata_df, load_cached_data=True, split_name="train"):
        """
        Generates features using vectorized merges.
        """
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(config.WORKING_DIR, f"features_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            if self.verbose:
                print(f"Loading features from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        if self.verbose:
            print(f"Generating features for {split_name} (Vectorized)...")

        # 1. Merge Atomic Coordinates and Properties for Atom 0 and Atom 1
        # Merge for Atom 0
        df = (
            metadata_df.merge(
                self.structures_df[
                    ["molecule_name", "atom_index", "x", "y", "z", "en", "rad"]
                ],
                left_on=["molecule_name", "atom_index_0"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            )
            .rename(
                columns={"x": "x0", "y": "y0", "z": "z0", "en": "en0", "rad": "rad0"}
            )
            .drop(columns=["atom_index"])
        )

        # Merge for Atom 1
        df = (
            df.merge(
                self.structures_df[
                    ["molecule_name", "atom_index", "x", "y", "z", "en", "rad"]
                ],
                left_on=["molecule_name", "atom_index_1"],
                right_on=["molecule_name", "atom_index"],
                how="left",
                suffixes=("", "_1"),  # Safety
            )
            .rename(
                columns={"x": "x1", "y": "y1", "z": "z1", "en": "en1", "rad": "rad1"}
            )
            .drop(columns=["atom_index"])
        )

        # 2. Geometric Features (Vectorized)
        df["dx"] = df["x0"] - df["x1"]
        df["dy"] = df["y0"] - df["y1"]
        df["dz"] = df["z0"] - df["z1"]
        df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)

        # Inverse Distance Features (Cite solution_lesson_node_00001)
        # Add epsilon to avoid div by zero (though atoms shouldn't overlap)
        epsilon = 1e-6
        df["dist_inv"] = 1 / (df["dist"] + epsilon)
        df["dist_inv2"] = 1 / (df["dist"] ** 2 + epsilon)
        df["dist_inv3"] = 1 / (df["dist"] ** 3 + epsilon)

        # 3. Merge Neighbor Features (L1 Counts)
        # Merge for Atom 0
        df = df.merge(
            self.neighbor_counts,
            left_on=["molecule_name", "atom_index_0"],
            right_index=True,
            how="left",
        )
        # Rename columns for Atom 0
        l1_cols = self.neighbor_counts.columns
        rename_map_0 = {c: f"a0_{c}" for c in l1_cols}
        df = df.rename(columns=rename_map_0)

        # Merge for Atom 1
        df = df.merge(
            self.neighbor_counts,
            left_on=["molecule_name", "atom_index_1"],
            right_index=True,
            how="left",
        )
        # Rename columns for Atom 1
        rename_map_1 = {c: f"a1_{c}" for c in l1_cols}
        df = df.rename(columns=rename_map_1)

        # Fill NaNs with 0 (for atoms with no neighbors found in the bond list)
        fill_cols = list(rename_map_0.values()) + list(rename_map_1.values())
        df[fill_cols] = df[fill_cols].fillna(0)

        # 4. Cleanup
        drop_cols = ["x0", "y0", "z0", "x1", "y1", "z1", "dx", "dy", "dz"]
        df = df.drop(columns=drop_cols)

        # Optimize memory
        df = utils.reduce_mem_usage(df, verbose=self.verbose)

        if self.verbose:
            print(f"Saving features to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

        return df
