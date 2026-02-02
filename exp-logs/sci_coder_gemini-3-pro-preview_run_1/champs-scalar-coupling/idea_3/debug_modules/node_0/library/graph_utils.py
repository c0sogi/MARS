import os
import numpy as np
import pandas as pd
import torch
from library.config import GASEConfig
from library.data_utils import load_structures, load_metadata


class MoleculeGraphBuilder:
    """
    Constructs graph representations of molecules for the MPNN.
    Handles node/edge feature engineering and efficient caching using Numpy/Parquet.
    """

    def __init__(self):
        self.atom_types = GASEConfig.ATOM_TYPES
        self.num_atom_types = len(self.atom_types)
        self.rbf_min = GASEConfig.MPNN_RBF_MIN
        self.rbf_max = GASEConfig.MPNN_RBF_MAX
        self.num_rbf = GASEConfig.MPNN_NUM_RBF
        self.gamma = 1.0 / ((self.rbf_max - self.rbf_min) / self.num_rbf) ** 2
        self.centers = np.linspace(self.rbf_min, self.rbf_max, self.num_rbf)

    def _get_cache_paths(self, split):
        """Returns file paths for cached data of a specific split."""
        cache_dir = GASEConfig.GRAPH_CACHE_DIR
        return {
            "nodes": os.path.join(cache_dir, f"{split}_nodes.npy"),
            "edge_indices": os.path.join(cache_dir, f"{split}_edge_indices.npy"),
            "edge_features": os.path.join(cache_dir, f"{split}_edge_features.npy"),
            "index": os.path.join(cache_dir, f"{split}_index.parquet"),
        }

    def _compute_rbf(self, distances):
        """
        Computes Radial Basis Function expansion of distances.
        Shape: (N_edges, num_rbf)
        """
        # distances shape: (N_edges, 1) or (N_edges,)
        if distances.ndim == 1:
            distances = distances[:, np.newaxis]

        # Broadcasting: (N_edges, 1) - (1, num_rbf) -> (N_edges, num_rbf)
        return np.exp(-self.gamma * (distances - self.centers[np.newaxis, :]) ** 2)

    def process_data(self, split, load_cached_data=True):
        """
        Main method to load, process, and cache graph data for a given split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            dict: Contains 'nodes', 'edge_indices', 'edge_features' (numpy arrays)
                  and 'index' (pandas DataFrame mapping molecule_name to array slices).
        """
        paths = self._get_cache_paths(split)

        # 1. Check Cache
        if load_cached_data:
            all_exist = all(os.path.exists(p) for p in paths.values())
            if all_exist:
                print(f"[{split}] Loading graph data from cache...")
                return {
                    "nodes": np.load(paths["nodes"]),
                    "edge_indices": np.load(paths["edge_indices"]),
                    "edge_features": np.load(paths["edge_features"]),
                    "index": pd.read_parquet(paths["index"]),
                }
            else:
                print(f"[{split}] Cache miss. Processing from scratch...")
        else:
            print(f"[{split}] Force processing from scratch...")

        # 2. Load Raw Data
        print(f"[{split}] Loading structures and metadata...")
        # Get list of molecules for this split
        meta_df = load_metadata(split)
        target_molecules = meta_df["molecule_name"].unique()

        # Load all structures and filter
        structures = load_structures()
        structures = structures[
            structures["molecule_name"].isin(target_molecules)
        ].copy()

        # Sort to ensure contiguous memory layout per molecule
        structures = structures.sort_values(
            ["molecule_name", "atom_index"]
        ).reset_index(drop=True)

        # 3. Generate Edges (Vectorized Radius Graph)
        print(
            f"[{split}] Computing neighbor lists (Radius={GASEConfig.GRAPH_RADIUS}A)..."
        )

        # Self-join to get all pairs within molecules
        # Optimization: Only join necessary columns
        struct_slim = structures[["molecule_name", "atom_index", "x", "y", "z", "atom"]]

        edges = pd.merge(
            struct_slim, struct_slim, on="molecule_name", suffixes=("_0", "_1")
        )

        # Calculate Distances
        d_sq = (
            (edges["x_0"] - edges["x_1"]) ** 2
            + (edges["y_0"] - edges["y_1"]) ** 2
            + (edges["z_0"] - edges["z_1"]) ** 2
        )
        edges["dist"] = np.sqrt(d_sq)

        # Filter: 0 < dist < Radius
        mask = (edges["dist"] > 0) & (edges["dist"] <= GASEConfig.GRAPH_RADIUS)
        edges = edges[mask].reset_index(drop=True)

        # 4. Compute Node Features
        print(f"[{split}] Computing node features...")

        # 4a. Atom Type Encoding (One-Hot)
        # Map atom strings to integers
        atom_map = {k: v for k, v in self.atom_types.items()}
        structures["atom_idx"] = structures["atom"].map(atom_map)

        # Create One-Hot
        num_atoms = len(structures)
        node_feats_type = np.zeros((num_atoms, self.num_atom_types), dtype=np.float32)
        node_feats_type[np.arange(num_atoms), structures["atom_idx"].values] = 1.0

        # 4b. Bag of Neighbors
        # We need to count neighbor types for each atom
        # Add atom type of neighbor (atom_1) to edges
        edges["atom_type_1"] = edges["atom_1"].map(atom_map)

        # Group by atom_0 (the source) and atom_type_1 (the neighbor type)
        # We use a pivot table or crosstab.
        # Note: We must ensure alignment with the original 'structures' dataframe.
        # We use (molecule_name, atom_index_0) as the key.

        # Create a unique ID for each atom in structures to map back easily
        # Since structures is sorted and reset, the dataframe index is the unique ID (global_atom_idx)
        # We need to map edges back to this global index.

        # Map (molecule, atom_index) -> global_index
        # Create a lookup series
        structures["global_idx"] = structures.index

        # Merge global indices into edges
        # Map src (atom_0) global index
        edges = (
            edges.merge(
                structures[["molecule_name", "atom_index", "global_idx"]],
                left_on=["molecule_name", "atom_index_0"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            )
            .rename(columns={"global_idx": "src_global_idx"})
            .drop(columns=["atom_index"])
        )

        # Map dst (atom_1) global index
        edges = (
            edges.merge(
                structures[["molecule_name", "atom_index", "global_idx"]],
                left_on=["molecule_name", "atom_index_1"],
                right_on=["molecule_name", "atom_index"],
                how="left",
            )
            .rename(columns={"global_idx": "dst_global_idx"})
            .drop(columns=["atom_index"])
        )

        # Now compute Bag of Neighbors using src_global_idx
        # Crosstab: rows=src_global_idx, cols=atom_type_1
        # This might be sparse, so we reindex to ensure all atoms are present
        bon_counts = pd.crosstab(edges["src_global_idx"], edges["atom_type_1"])

        # Reindex to include atoms with no neighbors (isolated) and ensure all columns (atom types) exist
        bon_counts = bon_counts.reindex(index=np.arange(num_atoms), fill_value=0)
        for t_idx in range(self.num_atom_types):
            if t_idx not in bon_counts.columns:
                bon_counts[t_idx] = 0
        bon_counts = bon_counts.sort_index(
            axis=1
        )  # Ensure column order matches type index

        node_feats_bon = bon_counts.values.astype(np.float32)

        # Combine Node Features
        final_node_feats = np.concatenate([node_feats_type, node_feats_bon], axis=1)

        # 5. Compute Edge Features
        print(f"[{split}] Computing edge features...")
        distances = edges["dist"].values

        # RBF
        rbf_feats = self._compute_rbf(distances)

        # Inverse Powers
        epsilon = 1e-6
        inv_1 = 1.0 / (distances + epsilon)
        inv_2 = 1.0 / (distances**2 + epsilon)
        inv_3 = 1.0 / (distances**3 + epsilon)
        inv_feats = np.stack([inv_1, inv_2, inv_3], axis=1)

        final_edge_feats = np.concatenate([rbf_feats, inv_feats], axis=1).astype(
            np.float32
        )

        # Edge Indices (src, dst)
        final_edge_indices = edges[["src_global_idx", "dst_global_idx"]].values.astype(
            np.int64
        )

        # 6. Create Index Map (Molecule -> Slices)
        print(f"[{split}] Building index map...")

        # Node slices: Group by molecule in structures
        # structures is already sorted by molecule_name
        mol_group = structures.groupby("molecule_name", sort=False)
        node_counts = mol_group.size()
        node_starts = np.concatenate([[0], np.cumsum(node_counts.values)[:-1]])

        # Edge slices: Group by molecule in edges
        # Edges might not be sorted by molecule_name perfectly after merge, so sort
        edges["mol_cat"] = pd.Categorical(
            edges["molecule_name"], categories=target_molecules, ordered=True
        )
        edges = edges.sort_values("mol_cat")

        # We need to ensure the edge arrays (features/indices) match this sorted order
        # Re-extract arrays after sort
        final_edge_feats = final_edge_feats[edges.index]  # Reorder using previous index
        final_edge_indices = edges[["src_global_idx", "dst_global_idx"]].values.astype(
            np.int64
        )

        edge_group = edges.groupby("molecule_name", sort=False, observed=True)
        edge_counts = edge_group.size()

        # Align edge counts with molecule list (some molecules might have 0 edges)
        edge_counts = edge_counts.reindex(target_molecules, fill_value=0)
        edge_starts = np.concatenate([[0], np.cumsum(edge_counts.values)[:-1]])

        index_df = pd.DataFrame(
            {
                "molecule_name": target_molecules,
                "node_start": node_starts,
                "node_count": node_counts.values,
                "edge_start": edge_starts,
                "edge_count": edge_counts.values,
            }
        )

        # 7. Save to Cache
        print(f"[{split}] Saving to {GASEConfig.GRAPH_CACHE_DIR}...")
        np.save(paths["nodes"], final_node_feats)
        np.save(paths["edge_indices"], final_edge_indices)
        np.save(paths["edge_features"], final_edge_feats)
        index_df.to_parquet(paths["index"], index=False)

        return {
            "nodes": final_node_feats,
            "edge_indices": final_edge_indices,
            "edge_features": final_edge_feats,
            "index": index_df,
        }
