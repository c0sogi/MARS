import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder

from library.config import (
    WORKING_DIR,
    BOND_LENGTH_THRESHOLD,
    ATOM_TYPES,
    TRAIN_PROCESSED_PATH,
    VAL_PROCESSED_PATH,
    TEST_PROCESSED_PATH,
)
from library.utils import reduce_mem_usage, Timer
from library.geometry_engine import GeometryEngine


class FeaturePipeline:
    def __init__(self, structures_path):
        """
        Initializes the FeaturePipeline.

        Args:
            structures_path (str): Path to the structures.csv file.
        """
        self.structures_path = structures_path
        self.geo_engine = GeometryEngine(structures_path)
        self.atom_neighbor_cache_path = os.path.join(
            WORKING_DIR, "atom_neighbors.parquet"
        )

    def _compute_all_atom_neighbors(self):
        """
        Computes the 'Bag of Neighbors' context for every atom in every molecule
        found in the structures file.

        Returns:
            pd.DataFrame: DataFrame with columns ['molecule_name', 'atom_index', 'n_H', 'n_C', 'n_N', 'n_O', 'n_F']
        """
        # Check cache first
        if os.path.exists(self.atom_neighbor_cache_path):
            print(
                f"Loading cached atom neighbor features from {self.atom_neighbor_cache_path}"
            )
            return pd.read_parquet(self.atom_neighbor_cache_path)

        print("Computing atom neighbor context features (Bag of Neighbors)...")

        # We will build a list of dicts
        neighbor_data = []

        # Iterate over all molecules in the geometry engine
        # structures_map is {molecule_name: {'coords': np.array, 'atoms': np.array}}
        for mol_name, data in self.geo_engine.structures_map.items():
            coords = data["coords"]
            atoms = data["atoms"]
            n_atoms = len(atoms)

            # Calculate distance matrix
            dists = squareform(pdist(coords))

            # Adjacency
            adj = dists < BOND_LENGTH_THRESHOLD
            np.fill_diagonal(adj, False)

            # Iterate over each atom to count neighbors
            for i in range(n_atoms):
                # Get indices of neighbors
                neighbor_indices = np.where(adj[i])[0]
                neighbor_atoms = atoms[neighbor_indices]

                # Count types
                counts = {f"n_{t}": 0 for t in ATOM_TYPES}
                unique, u_counts = np.unique(neighbor_atoms, return_counts=True)
                for u_atom, u_count in zip(unique, u_counts):
                    if u_atom in ATOM_TYPES:
                        counts[f"n_{u_atom}"] = u_count

                # Add metadata
                counts["molecule_name"] = mol_name
                counts["atom_index"] = i

                neighbor_data.append(counts)

        # Create DataFrame
        df_neighbors = pd.DataFrame(neighbor_data)

        # Optimize types
        df_neighbors = reduce_mem_usage(df_neighbors, verbose=False)

        # Save cache
        print(f"Saving atom neighbor features to {self.atom_neighbor_cache_path}")
        df_neighbors.to_parquet(self.atom_neighbor_cache_path, index=False)

        return df_neighbors

    def _add_distance_features(self, df):
        """
        Adds Euclidean distance and inverse distance power features.
        Assumes df has x0, y0, z0, x1, y1, z1 columns.
        """
        # Vectorized distance calculation
        d_x = df["x0"] - df["x1"]
        d_y = df["y0"] - df["y1"]
        d_z = df["z0"] - df["z1"]

        dist = np.sqrt(d_x**2 + d_y**2 + d_z**2)

        # Add features
        df["dist"] = dist
        df["dist_inv"] = 1.0 / np.maximum(dist, 1e-6)  # Avoid div by zero
        df["dist_inv2"] = df["dist_inv"] ** 2
        df["dist_inv3"] = df["dist_inv"] ** 3

        return df

    def generate_features(self, metadata_df, dataset_name, load_cached_data=True):
        """
        Main driver to generate the feature matrix.

        Args:
            metadata_df (pd.DataFrame): The metadata dataframe (train/val/test).
            dataset_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached parquet files.

        Returns:
            pd.DataFrame: The processed dataframe with features.
        """
        # Determine cache path
        if dataset_name == "train":
            cache_path = TRAIN_PROCESSED_PATH
        elif dataset_name == "val":
            cache_path = VAL_PROCESSED_PATH
        else:
            cache_path = TEST_PROCESSED_PATH

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{dataset_name}] Loading processed features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"[{dataset_name}] Generating features from scratch...")

        with Timer("Structure Merge"):
            # 2. Merge Structure Coordinates
            # We need coordinates for atom_0 and atom_1
            # structures.csv columns: molecule_name, atom_index, atom, x, y, z
            # We load the full structures df just for merging, or use the map if efficient.
            # Using pandas merge is easier for bulk operations.
            df_struct = pd.read_csv(self.structures_path)

            # Merge for Atom 0
            df = metadata_df.merge(
                df_struct[["molecule_name", "atom_index", "x", "y", "z", "atom"]],
                left_on=["molecule_name", "atom_index_0"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            )
            df = df.rename(columns={"x": "x0", "y": "y0", "z": "z0", "atom": "atom_0"})
            df = df.drop(columns=["atom_index"])

            # Merge for Atom 1
            df = df.merge(
                df_struct[["molecule_name", "atom_index", "x", "y", "z", "atom"]],
                left_on=["molecule_name", "atom_index_1"],
                right_on=["molecule_name", "atom_index"],
                how="left",
                suffixes=("", "_1"),
            )
            df = df.rename(columns={"x": "x1", "y": "y1", "z": "z1", "atom": "atom_1"})
            df = df.drop(columns=["atom_index"])

        with Timer("Distance Features"):
            # 3. Through-Space Features
            df = self._add_distance_features(df)

        with Timer("Geometry Engine (Cosine Features)"):
            # 4. Cosine Projection Features
            # Cite solution_lesson_node_00019
            df_geo = self.geo_engine.get_cosine_features(
                metadata_df, dataset_name, load_cached_data=load_cached_data
            )

            # Concatenate features
            df = df.reset_index(drop=True)
            df_geo = df_geo.reset_index(drop=True)
            df = pd.concat([df, df_geo], axis=1)

        with Timer("Node Context (Neighbors)"):
            # 5. Node-Context Features
            # Get pre-computed neighbor counts
            df_neighbors = self._compute_all_atom_neighbors()

            # Merge for Atom 0
            df = df.merge(
                df_neighbors,
                left_on=["molecule_name", "atom_index_0"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            ).drop(columns=["atom_index"])
            # Rename columns
            rename_dict_0 = {f"n_{t}": f"atom_0_n_{t}" for t in ATOM_TYPES}
            df = df.rename(columns=rename_dict_0)

            # Merge for Atom 1
            df = df.merge(
                df_neighbors,
                left_on=["molecule_name", "atom_index_1"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            ).drop(columns=["atom_index"])
            # Rename columns
            rename_dict_1 = {f"n_{t}": f"atom_1_n_{t}" for t in ATOM_TYPES}
            df = df.rename(columns=rename_dict_1)

        with Timer("Cleanup & Save"):
            # 6. Final Cleanup
            # Fill NaNs for cosine features
            cosine_cols = [c for c in df.columns if c.startswith("cos_")]
            df[cosine_cols] = df[cosine_cols].fillna(0)

            # Drop raw coordinates (rotational variance makes them noisy)
            df = df.drop(columns=["x0", "y0", "z0", "x1", "y1", "z1"])

            # Reduce memory
            df = reduce_mem_usage(df)

            # Save
            print(f"[{dataset_name}] Saving processed parquet to {cache_path}")
            df.to_parquet(cache_path)

        return df

    def prepare_data_for_type(self, df, coupling_type):
        """
        Prepares the feature matrix for a specific coupling type model.

        1. Filters by coupling_type.
        2. Drops constant columns (e.g., atom types for that coupling).
        3. Separates X (features) and y (target).

        Args:
            df (pd.DataFrame): The full feature dataframe.
            coupling_type (str): The specific type (e.g., '1JHC').

        Returns:
            tuple: (X, y) where X is a DataFrame and y is a Series (or None if test).
        """
        # Filter
        df_type = df[df["type"] == coupling_type].copy()

        if df_type.empty:
            return pd.DataFrame(), pd.Series()

        # Define columns to drop
        # 1. Metadata / Non-features
        cols_to_drop = ["id", "molecule_name", "type", "file_path"]

        # 2. Constant columns for specific types
        # For 1JHC, atom_0 is always H, atom_1 is always C (or vice versa depending on indexing).
        # We drop the string columns 'atom_0' and 'atom_1'.
        # We also drop the neighbor counts if they are constant, but that's risky.
        # We definitely drop the raw atom symbol columns.
        cols_to_drop.extend(["atom_0", "atom_1"])

        # Drop target if exists
        target = None
        if "scalar_coupling_constant" in df_type.columns:
            target = df_type["scalar_coupling_constant"]
            cols_to_drop.append("scalar_coupling_constant")

        # Drop columns
        # Use errors='ignore' in case some cols are missing
        X = df_type.drop(columns=cols_to_drop, errors="ignore")

        # Reset index
        X = X.reset_index(drop=True)
        if target is not None:
            target = target.reset_index(drop=True)

        return X, target
