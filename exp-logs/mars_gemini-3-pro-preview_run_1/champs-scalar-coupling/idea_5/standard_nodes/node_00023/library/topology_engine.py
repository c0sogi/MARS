import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from library import config, utils

# Disable chained assignment warning
pd.options.mode.chained_assignment = None


class TopologyEngine:
    """
    Vectorized Engine for Feature Engineering (Cite solution_lesson_node_00021).
    Replaces iterative molecule processing with global dataframe operations.
    """

    def __init__(self, structures_df, verbose=True):
        self.structures_df = structures_df
        self.verbose = verbose

    def _compute_bonds(self, molecules):
        """
        Computes all bonds for the given molecules using vectorized operations.
        Cite solution_lesson_node_00020: Use adaptive thresholds.
        """
        if self.verbose:
            print("Computing global bond network...")

        # Filter structures
        structs = self.structures_df[
            self.structures_df["molecule_name"].isin(molecules)
        ].copy()

        # Add covalent radii
        structs["rad"] = structs["atom"].map(config.COVALENT_RADII)

        # Self-join to get pairs
        bonds = structs.merge(structs, on="molecule_name", suffixes=("_i", "_j"))

        # Filter self-loops
        bonds = bonds[bonds["atom_index_i"] != bonds["atom_index_j"]]

        # Calculate distances
        bonds["dx"] = bonds["x_j"] - bonds["x_i"]
        bonds["dy"] = bonds["y_j"] - bonds["y_i"]
        bonds["dz"] = bonds["z_j"] - bonds["z_i"]
        bonds["dist"] = np.sqrt(bonds["dx"] ** 2 + bonds["dy"] ** 2 + bonds["dz"] ** 2)

        # Adaptive Threshold Filter
        threshold = bonds["rad_i"] + bonds["rad_j"] + config.BOND_TOLERANCE
        bonds = bonds[bonds["dist"] <= threshold]

        # Normalize vectors (for cosine features later)
        bonds["vec_x"] = bonds["dx"] / bonds["dist"]
        bonds["vec_y"] = bonds["dy"] / bonds["dist"]
        bonds["vec_z"] = bonds["dz"] / bonds["dist"]

        return bonds

    def generate_features(self, metadata_df, load_cached_data=True, split_name="train"):
        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(config.WORKING_DIR, f"features_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            if self.verbose:
                print(f"Loading features from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        if self.verbose:
            print(f"Generating features for {split_name} (Vectorized)...")

        # 1. Prepare Data
        unique_mols = metadata_df["molecule_name"].unique()
        bonds = self._compute_bonds(unique_mols)

        # 2. Node Features (Bag of Neighbors) - Cite solution_lesson_node_00006
        if self.verbose:
            print("Generating Node Features...")

        # Pivot to count neighbor types
        node_counts = pd.pivot_table(
            bonds,
            index=["molecule_name", "atom_index_i"],
            columns="atom_j",
            values="dist",
            aggfunc="count",
            fill_value=0,
        )
        node_counts.columns = [f"L1_{c}" for c in node_counts.columns]
        node_counts["n_bonds"] = node_counts.sum(axis=1)
        node_counts = node_counts.reset_index()

        # Merge Node Features to Metadata
        df = metadata_df.copy()

        # Merge for Atom 0
        df = (
            df.merge(
                node_counts,
                left_on=["molecule_name", "atom_index_0"],
                right_on=["molecule_name", "atom_index_i"],
                how="left",
            )
            .drop(columns=["atom_index_i"])
            .fillna(0)
        )
        df = df.rename(
            columns={
                c: f"a0_{c}"
                for c in node_counts.columns
                if c not in ["molecule_name", "atom_index_i"]
            }
        )

        # Merge for Atom 1
        df = (
            df.merge(
                node_counts,
                left_on=["molecule_name", "atom_index_1"],
                right_on=["molecule_name", "atom_index_i"],
                how="left",
            )
            .drop(columns=["atom_index_i"])
            .fillna(0)
        )
        df = df.rename(
            columns={
                c: f"a1_{c}"
                for c in node_counts.columns
                if c not in ["molecule_name", "atom_index_i"]
            }
        )

        # 3. Pair Geometric Features
        if self.verbose:
            print("Generating Pair Geometry...")

        # Add coordinates
        structs = self.structures_df[
            self.structures_df["molecule_name"].isin(unique_mols)
        ]
        structs_map = structs[["molecule_name", "atom_index", "x", "y", "z"]]

        df = (
            df.merge(
                structs_map,
                left_on=["molecule_name", "atom_index_0"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            )
            .rename(columns={"x": "x0", "y": "y0", "z": "z0"})
            .drop(columns=["atom_index"])
        )
        df = (
            df.merge(
                structs_map,
                left_on=["molecule_name", "atom_index_1"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            )
            .rename(columns={"x": "x1", "y": "y1", "z": "z1"})
            .drop(columns=["atom_index"])
        )

        # Calculate main vector and distance
        df["dx"] = df["x1"] - df["x0"]
        df["dy"] = df["y1"] - df["y0"]
        df["dz"] = df["z1"] - df["z0"]
        df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)
        df["dist_inv"] = 1.0 / df["dist"]
        df["dist_inv2"] = 1.0 / (df["dist"] ** 2)
        df["dist_inv3"] = 1.0 / (df["dist"] ** 3)

        # Normalize main vector (0 -> 1)
        df["u_x"] = df["dx"] / df["dist"]
        df["u_y"] = df["dy"] / df["dist"]
        df["u_z"] = df["dz"] / df["dist"]

        # 4. Angular Features (Cosine Projections) - Cite solution_lesson_node_00022
        if self.verbose:
            print("Generating Angular Features...")

        # We need to compute cosine of angle between bond(0->k) and bond(0->1)
        # Join df with bonds on atom 0
        # df has (id, mol, a0, a1, u_x, u_y, u_z)
        # bonds has (mol, a_i, vec_x, vec_y, vec_z)

        # Process Atom 0 Projections
        # Filter bonds to only relevant molecules to save memory
        bonds_rel = bonds[
            ["molecule_name", "atom_index_i", "vec_x", "vec_y", "vec_z"]
        ].copy()

        # Merge pairs with bonds for atom 0
        # Note: This expands the dataframe (one row per neighbor)
        merged_0 = df[
            ["id", "molecule_name", "atom_index_0", "u_x", "u_y", "u_z"]
        ].merge(
            bonds_rel,
            left_on=["molecule_name", "atom_index_0"],
            right_on=["molecule_name", "atom_index_i"],
            how="inner",
        )

        # Calculate Cosine: u . v
        merged_0["cos"] = (
            merged_0["u_x"] * merged_0["vec_x"]
            + merged_0["u_y"] * merged_0["vec_y"]
            + merged_0["u_z"] * merged_0["vec_z"]
        )

        # Aggregate
        agg_0 = (
            merged_0.groupby("id")["cos"]
            .agg(["mean", "min", "max"])
            .add_prefix("fp_0_")
        )
        df = df.merge(agg_0, on="id", how="left").fillna(0)

        # Process Atom 1 Projections
        # Vector 1->0 is -u
        merged_1 = df[
            ["id", "molecule_name", "atom_index_1", "u_x", "u_y", "u_z"]
        ].merge(
            bonds_rel,
            left_on=["molecule_name", "atom_index_1"],
            right_on=["molecule_name", "atom_index_i"],
            how="inner",
        )

        # Calculate Cosine: (-u) . v
        merged_1["cos"] = (
            -merged_1["u_x"] * merged_1["vec_x"]
            - merged_1["u_y"] * merged_1["vec_y"]
            - merged_1["u_z"] * merged_1["vec_z"]
        )

        agg_1 = (
            merged_1.groupby("id")["cos"]
            .agg(["mean", "min", "max"])
            .add_prefix("fp_1_")
        )
        df = df.merge(agg_1, on="id", how="left").fillna(0)

        # Cleanup intermediate columns
        drop_cols = [
            "x0",
            "y0",
            "z0",
            "x1",
            "y1",
            "z1",
            "dx",
            "dy",
            "dz",
            "u_x",
            "u_y",
            "u_z",
        ]
        df = df.drop(columns=drop_cols)

        # Optimize and Save
        df = utils.reduce_mem_usage(df, verbose=self.verbose)
        if self.verbose:
            print(f"Saving features to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

        return df
