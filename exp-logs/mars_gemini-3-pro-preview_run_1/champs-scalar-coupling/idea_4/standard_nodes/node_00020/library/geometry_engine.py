import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from library.config import BOND_LENGTH_THRESHOLD, WORKING_DIR
from library.utils import reduce_mem_usage


class GeometryEngine:
    def __init__(self, structures_path):
        """
        Initializes the GeometryEngine by loading and indexing molecular structures.

        Args:
            structures_path (str): Path to the structures.csv file.
        """
        self.structures_path = structures_path
        self.structures_map = self._load_structures()

    def _load_structures(self):
        """
        Loads structures and organizes them into a dictionary for fast lookup.
        Returns:
            dict: {molecule_name: {'coords': np.array, 'atoms': np.array}}
        """
        print(f"Loading structures from {self.structures_path}...")
        df = pd.read_csv(self.structures_path)

        # Sort by molecule_name and atom_index to ensure array indices match atom_index
        df = df.sort_values(["molecule_name", "atom_index"])

        structures_map = {}

        # Group by molecule_name for efficient access
        grouped = df.groupby("molecule_name")

        for name, group in grouped:
            coords = group[["x", "y", "z"]].values
            atoms = group["atom"].values
            # We assume atom_index corresponds to the position in the sorted group (0..N-1)
            structures_map[name] = {"coords": coords, "atoms": atoms}

        return structures_map

    def _build_graph(self, coords):
        """
        Builds a NetworkX graph for a molecule based on distance threshold.
        """
        dists = squareform(pdist(coords))

        # Create adjacency matrix based on bond length threshold
        adj_matrix = (dists < BOND_LENGTH_THRESHOLD).astype(int)
        np.fill_diagonal(adj_matrix, 0)

        # Create graph
        G = nx.from_numpy_array(adj_matrix)
        return G

    def get_cosine_features(self, df, dataset_name, load_cached_data=True):
        """
        Computes cosine projection features (Aggregate Directional Statistics).
        Cite solution_lesson_node_00019

        Args:
            df (pd.DataFrame): Dataframe containing 'molecule_name', 'atom_index_0', 'atom_index_1'.
            dataset_name (str): Name of the dataset for caching.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            pd.DataFrame: Dataframe with cosine features.
        """
        os.makedirs(WORKING_DIR, exist_ok=True)
        cache_file = os.path.join(
            WORKING_DIR, f"cosine_features_{dataset_name}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached cosine features from {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"Computing cosine features for {dataset_name}...")

        n_samples = len(df)
        # Features: mean, min, max for atom 0 and atom 1
        # Initialize with NaNs
        res = {
            "cos_0_mean": np.full(n_samples, np.nan, dtype=np.float32),
            "cos_0_min": np.full(n_samples, np.nan, dtype=np.float32),
            "cos_0_max": np.full(n_samples, np.nan, dtype=np.float32),
            "cos_1_mean": np.full(n_samples, np.nan, dtype=np.float32),
            "cos_1_min": np.full(n_samples, np.nan, dtype=np.float32),
            "cos_1_max": np.full(n_samples, np.nan, dtype=np.float32),
        }

        df_temp = df.copy()
        df_temp["temp_row_idx"] = np.arange(n_samples)
        grouped = df_temp.groupby("molecule_name")

        for mol_name, group in grouped:
            if mol_name not in self.structures_map:
                continue

            mol_data = self.structures_map[mol_name]
            coords = mol_data["coords"]
            n_atoms = len(coords)

            # Build Adjacency (Neighbors)
            dists = squareform(pdist(coords))
            adj = dists < BOND_LENGTH_THRESHOLD
            np.fill_diagonal(adj, False)

            # Pre-compute neighbors list for this molecule
            neighbors_list = [np.where(adj[i])[0] for i in range(n_atoms)]

            # Process rows
            for row in group.itertuples():
                idx0 = row.atom_index_0
                idx1 = row.atom_index_1
                row_idx = row.temp_row_idx

                p0 = coords[idx0]
                p1 = coords[idx1]

                # Axis vectors (pointing towards the other atom)
                axis_0 = p1 - p0
                axis_1 = p0 - p1

                norm_axis_0 = np.linalg.norm(axis_0)
                norm_axis_1 = np.linalg.norm(axis_1)

                # Avoid zero division
                if norm_axis_0 < 1e-6:
                    norm_axis_0 = 1.0
                if norm_axis_1 < 1e-6:
                    norm_axis_1 = 1.0

                # --- Atom 0 ---
                neighs_0 = neighbors_list[idx0]
                if len(neighs_0) > 0:
                    # Vectors to neighbors
                    vecs = coords[neighs_0] - p0
                    norms = np.linalg.norm(vecs, axis=1)
                    norms[norms < 1e-6] = 1.0

                    # Dot products
                    dots = np.dot(vecs, axis_0)
                    cosines = dots / (norms * norm_axis_0)

                    # Clip
                    cosines = np.clip(cosines, -1.0, 1.0)

                    res["cos_0_mean"][row_idx] = np.mean(cosines)
                    res["cos_0_min"][row_idx] = np.min(cosines)
                    res["cos_0_max"][row_idx] = np.max(cosines)
                else:
                    res["cos_0_mean"][row_idx] = 0.0
                    res["cos_0_min"][row_idx] = 0.0
                    res["cos_0_max"][row_idx] = 0.0

                # --- Atom 1 ---
                neighs_1 = neighbors_list[idx1]
                if len(neighs_1) > 0:
                    vecs = coords[neighs_1] - p1
                    norms = np.linalg.norm(vecs, axis=1)
                    norms[norms < 1e-6] = 1.0

                    dots = np.dot(vecs, axis_1)
                    cosines = dots / (norms * norm_axis_1)

                    cosines = np.clip(cosines, -1.0, 1.0)

                    res["cos_1_mean"][row_idx] = np.mean(cosines)
                    res["cos_1_min"][row_idx] = np.min(cosines)
                    res["cos_1_max"][row_idx] = np.max(cosines)
                else:
                    res["cos_1_mean"][row_idx] = 0.0
                    res["cos_1_min"][row_idx] = 0.0
                    res["cos_1_max"][row_idx] = 0.0

        result_df = pd.DataFrame(res)
        result_df = reduce_mem_usage(result_df, verbose=False)

        print(f"Saving cosine features to {cache_file}")
        result_df.to_parquet(cache_file)
        return result_df
