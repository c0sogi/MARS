import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# 1. Graph Processing & Caching
# ==========================================


def process_and_cache_graphs(structures_path, cache_dir, load_cached_data=True):
    """
    Processes molecular structures into graph data (nodes, edges, triplets)
    and caches them as numpy arrays.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # File paths for cache
    paths = {
        "nodes": os.path.join(cache_dir, "nodes.npy"),
        "coords": os.path.join(cache_dir, "coords.npy"),
        "edges": os.path.join(cache_dir, "edges.npy"),
        "triplets": os.path.join(cache_dir, "triplets.npy"),
        "index": os.path.join(cache_dir, "mol_index.parquet"),
    }

    # Check if cache exists
    if load_cached_data and all(os.path.exists(p) for p in paths.values()):
        print(f"Loading cached graph data from {cache_dir}...")
        return paths

    print("Processing structures and generating graph cache...")

    # Load structures
    df_struct = pd.read_csv(structures_path)
    # Ensure sorted by molecule and atom index for consistent indexing
    df_struct = df_struct.sort_values(["molecule_name", "atom_index"]).reset_index(
        drop=True
    )

    # Map atoms to integers
    atom_map = Config.ATOM_MAP
    df_struct["atom_type"] = df_struct["atom"].map(atom_map).astype(np.int32)

    # Group by molecule
    grouped = df_struct.groupby("molecule_name")

    # Lists to collect data
    all_nodes = []
    all_coords = []
    all_edges = []
    all_triplets = []

    # Index data
    mol_names = []
    node_starts = []
    node_counts = []
    edge_starts = []
    edge_counts = []
    triplet_starts = []
    triplet_counts = []

    current_node_start = 0
    current_edge_start = 0
    current_triplet_start = 0

    cutoff = Config.CUTOFF

    # Iterate over molecules
    # Note: Using a loop is feasible for ~60k-80k molecules if operations inside are efficient
    for mol_name, group in grouped:
        # 1. Nodes & Coords
        coords = group[["x", "y", "z"]].values.astype(np.float32)
        types = group["atom_type"].values.astype(np.int32)
        num_atoms = len(types)

        all_nodes.append(types)
        all_coords.append(coords)

        # 2. Edges (Distance < Cutoff)
        # Compute distance matrix
        # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        # Find pairs within cutoff (excluding self-loops)
        # np.where returns (row_indices, col_indices) -> (j, i)
        # We want edges j -> i
        mask = (dists <= cutoff) & (dists > 0)  # Exclude self
        src, dst = np.where(mask)

        num_edges = len(src)
        edge_indices = np.stack([src, dst], axis=0).T.astype(np.int32)  # (E, 2)

        all_edges.append(edge_indices)

        # 3. Triplets (k -> j -> i)
        # We need pairs of edges (e_kj, e_ji) such that dst(e_kj) == src(e_ji)
        # and src(e_kj) != dst(e_ji) (k != i)

        if num_edges > 0:
            # Create a dataframe for efficient join
            # This is locally small (max ~30 atoms, ~900 edges), so pandas overhead is acceptable
            # or we can use numpy broadcasting.
            # Given small N, numpy broadcasting is faster.

            # e_kj: indices [0, ..., E-1]
            # e_ji: indices [0, ..., E-1]

            # We want to match edge_indices[e_kj, 1] == edge_indices[e_ji, 0]

            # dst of all edges
            dst_all = edge_indices[:, 1]  # (E,)
            # src of all edges
            src_all = edge_indices[:, 0]  # (E,)

            # Broadcast comparison: (E, 1) == (1, E) -> (E, E) matrix
            # rows are e_kj, cols are e_ji
            # connectivity[k, m] is True if edge k connects to edge m
            connectivity = dst_all[:, np.newaxis] == src_all[np.newaxis, :]

            # Also ensure k != i (no backtracking immediately, though physically valid, usually excluded or handled by angle)
            # src(e_kj) != dst(e_ji)
            src_kj = edge_indices[:, 0]
            dst_ji = edge_indices[:, 1]
            non_backtrack = src_kj[:, np.newaxis] != dst_ji[np.newaxis, :]

            valid_triplets = connectivity & non_backtrack

            # Get indices of True values
            idx_kj, idx_ji = np.where(valid_triplets)

            triplet_indices = np.stack([idx_kj, idx_ji], axis=0).T.astype(np.int32)
        else:
            triplet_indices = np.zeros((0, 2), dtype=np.int32)

        num_triplets = len(triplet_indices)
        all_triplets.append(triplet_indices)

        # Update Index Map
        mol_names.append(mol_name)

        node_starts.append(current_node_start)
        node_counts.append(num_atoms)
        current_node_start += num_atoms

        edge_starts.append(current_edge_start)
        edge_counts.append(num_edges)
        current_edge_start += num_edges

        triplet_starts.append(current_triplet_start)
        triplet_counts.append(num_triplets)
        current_triplet_start += num_triplets

    # Concatenate and Save
    print("Concatenating and saving arrays...")

    np.save(paths["nodes"], np.concatenate(all_nodes))
    np.save(paths["coords"], np.concatenate(all_coords))

    # Handle case where no edges/triplets exist in entire dataset (unlikely but safe)
    if all_edges:
        np.save(paths["edges"], np.concatenate(all_edges))
    else:
        np.save(paths["edges"], np.zeros((0, 2), dtype=np.int32))

    if all_triplets:
        np.save(paths["triplets"], np.concatenate(all_triplets))
    else:
        np.save(paths["triplets"], np.zeros((0, 2), dtype=np.int32))

    # Save Index Map
    df_index = pd.DataFrame(
        {
            "molecule_name": mol_names,
            "node_start": node_starts,
            "node_count": node_counts,
            "edge_start": edge_starts,
            "edge_count": edge_counts,
            "triplet_start": triplet_starts,
            "triplet_count": triplet_counts,
        }
    )
    df_index.to_parquet(paths["index"], index=False)

    print("Graph processing complete.")
    return paths


# ==========================================
# 2. Dataset Class
# ==========================================


class MoleculeDataset(Dataset):
    def __init__(
        self, metadata_path, cache_paths, mode="train", debug_sample_size=None
    ):
        """
        Args:
            metadata_path: Path to train/val/test metadata csv.
            cache_paths: Dict containing paths to cached npy/parquet files.
            mode: 'train', 'val', or 'test'.
            debug_sample_size: If set, limit dataset size for debugging.
        """
        self.mode = mode

        # 1. Load Metadata
        df_meta = pd.read_csv(metadata_path)

        if debug_sample_size:
            # Filter for a subset of molecules
            unique_mols = df_meta["molecule_name"].unique()
            subset_mols = unique_mols[:debug_sample_size]
            df_meta = df_meta[df_meta["molecule_name"].isin(subset_mols)].copy()
            print(f"Debug Mode: Sampled {len(subset_mols)} molecules.")

        # Group targets by molecule for efficient retrieval
        # We want to retrieve all targets for a molecule at once
        self.mol_groups = {k: v for k, v in df_meta.groupby("molecule_name")}
        self.mol_names = list(self.mol_groups.keys())

        # 2. Load Graph Cache
        self.nodes = np.load(cache_paths["nodes"], mmap_mode="r")
        self.coords = np.load(cache_paths["coords"], mmap_mode="r")
        self.edges = np.load(cache_paths["edges"], mmap_mode="r")
        self.triplets = np.load(cache_paths["triplets"], mmap_mode="r")

        df_index = pd.read_parquet(cache_paths["index"])
        self.mol_index = df_index.set_index("molecule_name")

        # Pre-compute coupling type map
        self.coupling_map = Config.COUPLING_MAP

    def __len__(self):
        return len(self.mol_names)

    def __getitem__(self, idx):
        mol_name = self.mol_names[idx]

        # 1. Retrieve Graph Data
        try:
            info = self.mol_index.loc[mol_name]
        except KeyError:
            # Should not happen if metadata and structures are consistent
            raise KeyError(f"Molecule {mol_name} not found in structure cache.")

        # Slice arrays
        # Use .copy() to convert from mmap to in-memory array for current sample
        z = self.nodes[
            info["node_start"] : info["node_start"] + info["node_count"]
        ].copy()
        pos = self.coords[
            info["node_start"] : info["node_start"] + info["node_count"]
        ].copy()

        edge_index = self.edges[
            info["edge_start"] : info["edge_start"] + info["edge_count"]
        ].copy()
        # edge_index is (E, 2), transpose to (2, E) for PyG convention
        edge_index = edge_index.T

        triplet_indices = self.triplets[
            info["triplet_start"] : info["triplet_start"] + info["triplet_count"]
        ].copy()
        # triplet_indices is (T, 2) -> col 0 is idx_kj, col 1 is idx_ji
        if len(triplet_indices) > 0:
            idx_kj = triplet_indices[:, 0]
            idx_ji = triplet_indices[:, 1]
        else:
            idx_kj = np.array([], dtype=np.int32)
            idx_ji = np.array([], dtype=np.int32)

        # 2. Retrieve Targets
        df_targets = self.mol_groups[mol_name]

        target_node_0 = df_targets["atom_index_0"].values.astype(np.int64)
        target_node_1 = df_targets["atom_index_1"].values.astype(np.int64)

        # Map types
        target_type = df_targets["type"].map(self.coupling_map).values.astype(np.int64)

        # Target values (if available)
        if "scalar_coupling_constant" in df_targets.columns:
            target_val = df_targets["scalar_coupling_constant"].values.astype(
                np.float32
            )
        else:
            target_val = np.zeros(len(target_node_0), dtype=np.float32)

        # IDs for submission
        target_ids = df_targets["id"].values.astype(np.int64)

        # 3. Map Target Pairs to Graph Edges
        # We need to find the edge index for u->v and v->u
        # Create a lookup map: (u, v) -> edge_idx
        # Since E is small (~100-1000), a dict is fast enough

        # edge_index is (2, E)
        edge_lookup = {}
        for e_idx in range(edge_index.shape[1]):
            u, v = edge_index[0, e_idx], edge_index[1, e_idx]
            edge_lookup[(u, v)] = e_idx

        target_edge_index_uv = []
        target_edge_index_vu = []

        for u, v in zip(target_node_0, target_node_1):
            # u -> v
            idx_uv = edge_lookup.get((u, v), -1)
            target_edge_index_uv.append(idx_uv)

            # v -> u
            idx_vu = edge_lookup.get((v, u), -1)
            target_edge_index_vu.append(idx_vu)

        target_edge_index_uv = np.array(target_edge_index_uv, dtype=np.int64)
        target_edge_index_vu = np.array(target_edge_index_vu, dtype=np.int64)

        return {
            "z": torch.from_numpy(z).long(),
            "pos": torch.from_numpy(pos).float(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "idx_kj": torch.from_numpy(idx_kj).long(),
            "idx_ji": torch.from_numpy(idx_ji).long(),
            "target_node_0": torch.from_numpy(target_node_0).long(),
            "target_node_1": torch.from_numpy(target_node_1).long(),
            "target_type": torch.from_numpy(target_type).long(),
            "target_val": torch.from_numpy(target_val).float(),
            "target_ids": torch.from_numpy(target_ids).long(),
            "target_edge_index_uv": torch.from_numpy(target_edge_index_uv).long(),
            "target_edge_index_vu": torch.from_numpy(target_edge_index_vu).long(),
            "num_nodes": len(z),
            "num_edges": edge_index.shape[1],
        }


# ==========================================
# 3. Collate Function
# ==========================================


def collate_graphs(batch):
    """
    Batches a list of graph dictionaries into a single dictionary.
    Adjusts indices (edge_index, triplets, target_nodes) by cumulative counts.
    """
    # Initialize lists
    z_list = []
    pos_list = []
    edge_index_list = []
    idx_kj_list = []
    idx_ji_list = []

    target_node_0_list = []
    target_node_1_list = []
    target_type_list = []
    target_val_list = []
    target_ids_list = []
    target_edge_index_uv_list = []
    target_edge_index_vu_list = []

    batch_idx_list = (
        []
    )  # Map atoms to batch index if needed (not strictly used by model but good practice)

    cumsum_nodes = 0
    cumsum_edges = 0

    for i, data in enumerate(batch):
        num_nodes = data["num_nodes"]
        num_edges = data["num_edges"]

        # 1. Nodes & Pos
        z_list.append(data["z"])
        pos_list.append(data["pos"])
        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))

        # 2. Edges
        # Offset node indices in edge_index
        edge_index_list.append(data["edge_index"] + cumsum_nodes)

        # 3. Triplets
        # Offset edge indices in triplets
        idx_kj_list.append(data["idx_kj"] + cumsum_edges)
        idx_ji_list.append(data["idx_ji"] + cumsum_edges)

        # 4. Targets
        # Offset node indices
        target_node_0_list.append(data["target_node_0"] + cumsum_nodes)
        target_node_1_list.append(data["target_node_1"] + cumsum_nodes)

        target_type_list.append(data["target_type"])
        target_val_list.append(data["target_val"])
        target_ids_list.append(data["target_ids"])

        # Offset edge indices for targets
        # Only offset valid indices (>= 0)
        t_uv = data["target_edge_index_uv"].clone()
        mask_uv = t_uv >= 0
        t_uv[mask_uv] += cumsum_edges
        target_edge_index_uv_list.append(t_uv)

        t_vu = data["target_edge_index_vu"].clone()
        mask_vu = t_vu >= 0
        t_vu[mask_vu] += cumsum_edges
        target_edge_index_vu_list.append(t_vu)

        # Update cumsums
        cumsum_nodes += num_nodes
        cumsum_edges += num_edges

    # Concatenate
    batch_data = {
        "z": torch.cat(z_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),  # (2, Total_Edges)
        "idx_kj": torch.cat(idx_kj_list, dim=0),
        "idx_ji": torch.cat(idx_ji_list, dim=0),
        "target_node_0": torch.cat(target_node_0_list, dim=0),
        "target_node_1": torch.cat(target_node_1_list, dim=0),
        "target_type": torch.cat(target_type_list, dim=0),
        "target_val": torch.cat(target_val_list, dim=0),
        "target_ids": torch.cat(target_ids_list, dim=0),
        "target_edge_index_uv": torch.cat(target_edge_index_uv_list, dim=0),
        "target_edge_index_vu": torch.cat(target_edge_index_vu_list, dim=0),
        "batch": torch.cat(batch_idx_list, dim=0),
    }

    return batch_data


# ==========================================
# 4. Data Loaders Factory
# ==========================================


def get_dataloaders(
    train_meta_path,
    val_meta_path,
    test_meta_path,
    structures_path,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    load_cached_data=True,
):
    """
    Prepares and returns DataLoaders for train, val, and test sets.
    """
    # 1. Process and Cache Graphs
    cache_paths = process_and_cache_graphs(
        structures_path, Config.WORKING_DIR, load_cached_data=load_cached_data
    )

    # 2. Create Datasets
    train_dataset = MoleculeDataset(
        train_meta_path, cache_paths, mode="train", debug_sample_size=debug_sample_size
    )

    val_dataset = MoleculeDataset(
        val_meta_path, cache_paths, mode="val", debug_sample_size=debug_sample_size
    )

    test_dataset = MoleculeDataset(
        test_meta_path, cache_paths, mode="test", debug_sample_size=debug_sample_size
    )

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
