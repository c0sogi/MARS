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

    def _calculate_angle(self, p0, p1, p2):
        """
        Calculates the angle (in degrees) at p1 formed by p0-p1-p2.
        """
        v1 = p0 - p1
        v2 = p2 - p1

        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-6 or n2 < 1e-6:
            return np.nan

        v1_u = v1 / n1
        v2_u = v2 / n2

        cosine_angle = np.dot(v1_u, v2_u)
        # Clip to handle floating point errors slightly outside [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def _calculate_dihedral(self, p0, p1, p2, p3):
        """
        Calculates the dihedral angle (in degrees) for the chain p0-p1-p2-p3.
        """
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize b1
        b1_norm = np.linalg.norm(b1)
        if b1_norm < 1e-6:
            return np.nan
        b1_u = b1 / b1_norm

        # Vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        v = b0 - np.dot(b0, b1_u) * b1_u
        w = b2 - np.dot(b2, b1_u) * b1_u

        # Angle between v and w
        x = np.dot(v, w)
        y = np.dot(np.cross(b1_u, v), w)

        return np.degrees(np.arctan2(y, x))

    def get_shortest_path_features(self, df, dataset_name, load_cached_data=True):
        """
        Computes geometric path features for the given dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing 'molecule_name', 'atom_index_0', 'atom_index_1'.
            dataset_name (str): Name of the dataset (e.g., 'train', 'test') for caching.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: Dataframe with geometric features, aligned with input df.
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)
        cache_file = os.path.join(
            WORKING_DIR, f"geometry_features_{dataset_name}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached geometry features from {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"Computing geometry features for {dataset_name}...")

        # Prepare results container
        n_samples = len(df)
        path_lengths = np.full(n_samples, -1, dtype=np.int8)
        path_angles = np.full(n_samples, np.nan, dtype=np.float32)
        path_dihedrals = np.full(n_samples, np.nan, dtype=np.float32)

        # Add a temp index to df to track order for filling arrays
        df_temp = df.copy()
        df_temp["temp_row_idx"] = np.arange(n_samples)

        # Group by molecule to process efficiently
        grouped = df_temp.groupby("molecule_name")

        # Iterate over molecules
        for mol_name, group in grouped:
            if mol_name not in self.structures_map:
                continue

            mol_data = self.structures_map[mol_name]
            coords = mol_data["coords"]

            # Build Graph
            G = self._build_graph(coords)

            # Compute all pairs shortest paths (BFS)
            # path_dict[source][target] = [list of nodes]
            try:
                path_dict = dict(nx.all_pairs_shortest_path(G))
            except Exception:
                # Fallback for empty/error graphs
                path_dict = {}

            # Process rows for this molecule
            for row in group.itertuples():
                idx0 = row.atom_index_0
                idx1 = row.atom_index_1
                row_idx = row.temp_row_idx

                if idx0 in path_dict and idx1 in path_dict[idx0]:
                    path = path_dict[idx0][idx1]
                    plen = len(path) - 1  # Number of bonds

                    path_lengths[row_idx] = plen

                    if plen == 2:
                        # 2J: Angle at path[1] (A-B-C)
                        angle = self._calculate_angle(
                            coords[path[0]], coords[path[1]], coords[path[2]]
                        )
                        path_angles[row_idx] = angle
                    elif plen == 3:
                        # 3J: Dihedral (A-B-C-D)
                        dihedral = self._calculate_dihedral(
                            coords[path[0]],
                            coords[path[1]],
                            coords[path[2]],
                            coords[path[3]],
                        )
                        path_dihedrals[row_idx] = dihedral
                else:
                    # No path found (disconnected or error)
                    path_lengths[row_idx] = -1

        # Construct Result DataFrame
        result_df = pd.DataFrame(
            {
                "geo_path_len": path_lengths,
                "geo_angle": path_angles,
                "geo_dihedral": path_dihedrals,
            }
        )

        # Reduce memory usage
        result_df = reduce_mem_usage(result_df, verbose=False)

        # Save to cache
        print(f"Saving geometry features to {cache_file}")
        result_df.to_parquet(cache_file)

        return result_df
