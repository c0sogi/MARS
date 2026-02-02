import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import ase.io
from ase.neighborlist import neighbor_list
from library.config import Config


def process_structure(file_path, cutoff):
    """
    Reads an XYZ file and constructs a neighbor graph.

    Args:
        file_path (str): Path to the .xyz file relative to input directory.
        cutoff (float): Cutoff radius for neighbor search.

    Returns:
        dict: Contains 'atomic_numbers', 'edge_index', 'edge_distances'.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    try:
        # Cite debug_lesson_1: Explicitly Define Parsers When File Extensions Are Misleading
        atoms = ase.io.read(full_path, format="aims")
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None

    # Get atomic numbers (node features)
    atomic_numbers = atoms.get_atomic_numbers()

    # Compute neighbors with PBC
    # i: source indices, j: target indices, d: distances
    # self_interaction=False to exclude self-loops
    i, j, d = neighbor_list("ijd", atoms, cutoff, self_interaction=False)

    # edge_index: [2, num_edges]
    edge_index = np.vstack((i, j))

    return {
        "atomic_numbers": atomic_numbers,  # (num_nodes,)
        "edge_index": edge_index,  # (2, num_edges)
        "edge_distances": d,  # (num_edges,)
    }


def save_graphs_to_cache(graphs, targets, ids, cache_path):
    """
    Saves graph data to a .npz file without using pickle.
    """
    # Flatten lists
    all_atomic_numbers = []
    all_edge_src = []
    all_edge_dst = []
    all_edge_distances = []

    nodes_per_graph = []
    edges_per_graph = []

    for g in graphs:
        all_atomic_numbers.append(g["atomic_numbers"])
        # Check if edge_index is empty
        if g["edge_index"].shape[1] > 0:
            all_edge_src.append(g["edge_index"][0])
            all_edge_dst.append(g["edge_index"][1])
            all_edge_distances.append(g["edge_distances"])
        else:
            # Handle graphs with no edges (rare but possible)
            pass

        nodes_per_graph.append(len(g["atomic_numbers"]))
        edges_per_graph.append(len(g["edge_distances"]))

    # Concatenate
    if len(all_atomic_numbers) > 0:
        cat_atomic_numbers = np.concatenate(all_atomic_numbers)
    else:
        cat_atomic_numbers = np.array([], dtype=np.int64)

    if len(all_edge_src) > 0:
        cat_edge_src = np.concatenate(all_edge_src)
        cat_edge_dst = np.concatenate(all_edge_dst)
        cat_edge_distances = np.concatenate(all_edge_distances)
    else:
        cat_edge_src = np.array([], dtype=np.int64)
        cat_edge_dst = np.array([], dtype=np.int64)
        cat_edge_distances = np.array([], dtype=np.float32)

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    np.savez(
        cache_path,
        atomic_numbers=cat_atomic_numbers,
        edge_src=cat_edge_src,
        edge_dst=cat_edge_dst,
        edge_distances=cat_edge_distances,
        nodes_per_graph=np.array(nodes_per_graph, dtype=np.int32),
        edges_per_graph=np.array(edges_per_graph, dtype=np.int32),
        targets=np.array(targets, dtype=np.float32),
        ids=np.array(ids, dtype=np.int32),
    )
    print(f"Saved cache to {cache_path}")


def load_graphs_from_cache(cache_path):
    """
    Loads graph data from a .npz file.
    """
    print(f"Loading cache from {cache_path}...")
    data = np.load(cache_path)

    cat_atomic_numbers = data["atomic_numbers"]
    cat_edge_src = data["edge_src"]
    cat_edge_dst = data["edge_dst"]
    cat_edge_distances = data["edge_distances"]
    nodes_per_graph = data["nodes_per_graph"]
    edges_per_graph = data["edges_per_graph"]
    targets = data["targets"]
    ids = data["ids"]

    graphs = []

    # Reconstruct list
    node_offset = 0
    edge_offset = 0

    for i in range(len(nodes_per_graph)):
        n_nodes = nodes_per_graph[i]
        n_edges = edges_per_graph[i]

        g_atoms = cat_atomic_numbers[node_offset : node_offset + n_nodes]

        if n_edges > 0:
            g_src = cat_edge_src[edge_offset : edge_offset + n_edges]
            g_dst = cat_edge_dst[edge_offset : edge_offset + n_edges]
            g_dists = cat_edge_distances[edge_offset : edge_offset + n_edges]
            edge_index = np.vstack((g_src, g_dst))
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            g_dists = np.array([], dtype=np.float32)

        graphs.append(
            {
                "atomic_numbers": g_atoms,
                "edge_index": edge_index,
                "edge_distances": g_dists,
            }
        )

        node_offset += n_nodes
        edge_offset += n_edges

    return graphs, targets, ids


def get_data(
    metadata_path, split_name, load_cached_data=True, debug=False, debug_size=100
):
    """
    Main function to get data for a split (train/val/test).
    Handles caching logic.
    """
    # Determine cache filename
    suffix = "_debug" if debug else ""
    cache_filename = f"{split_name}_graphs{suffix}.npz"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            graphs, targets, ids = load_graphs_from_cache(cache_path)
            print(f"Loaded {len(graphs)} graphs from cache.")
            return graphs, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split_name} data from scratch...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if debug:
        df = df.iloc[:debug_size]
        print(f"Debug mode: using {len(df)} samples.")

    graphs = []
    targets = []
    ids = []

    for idx, row in df.iterrows():
        # Process graph
        g = process_structure(row["file_path"], Config.CUTOFF_RADIUS)
        if g is None:
            continue

        graphs.append(g)
        ids.append(row["id"])

        # Get targets if available (train/val)
        # Check if target columns exist in row
        row_targets = []
        has_targets = True
        for col in Config.TARGET_COLS:
            if col not in row:
                has_targets = False
                break
            row_targets.append(row[col])

        if has_targets:
            targets.append(row_targets)
        else:
            # For test set, fill with NaNs
            targets.append([np.nan] * len(Config.TARGET_COLS))

    # 3. Save to cache
    save_graphs_to_cache(graphs, targets, ids, cache_path)

    return graphs, targets, ids


class CrystalGraphDataset(Dataset):
    def __init__(self, graphs, targets, ids):
        self.graphs = graphs
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y = self.targets[idx]
        mid = self.ids[idx]

        return {
            "atomic_numbers": torch.tensor(g["atomic_numbers"], dtype=torch.long),
            "edge_index": torch.tensor(g["edge_index"], dtype=torch.long),
            "edge_distances": torch.tensor(g["edge_distances"], dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
            "id": mid,
        }


def collate_graphs(batch):
    """
    Collates a list of graph dictionaries into a single batch.
    """
    # batch is a list of dicts from __getitem__

    all_atomic_numbers = []
    all_edge_indices = []
    all_edge_distances = []
    all_targets = []
    all_ids = []
    batch_indices = []  # Maps nodes to graph index in batch

    node_offset = 0

    for i, item in enumerate(batch):
        num_nodes = item["atomic_numbers"].shape[0]

        all_atomic_numbers.append(item["atomic_numbers"])

        # Offset edge indices
        edge_index = item["edge_index"] + node_offset
        all_edge_indices.append(edge_index)

        all_edge_distances.append(item["edge_distances"])
        all_targets.append(item["target"])
        all_ids.append(item["id"])

        # Batch index for nodes
        batch_indices.append(torch.full((num_nodes,), i, dtype=torch.long))

        node_offset += num_nodes

    # Concatenate
    cat_atomic_numbers = torch.cat(all_atomic_numbers, dim=0)

    if len(all_edge_indices) > 0:
        cat_edge_indices = torch.cat(all_edge_indices, dim=1)  # (2, total_edges)
        cat_edge_distances = torch.cat(all_edge_distances, dim=0)
    else:
        cat_edge_indices = torch.empty((2, 0), dtype=torch.long)
        cat_edge_distances = torch.empty((0,), dtype=torch.float32)

    cat_targets = torch.stack(all_targets, dim=0)
    cat_batch = torch.cat(batch_indices, dim=0)

    return {
        "x": cat_atomic_numbers,
        "edge_index": cat_edge_indices,
        "edge_attr": cat_edge_distances,  # In this model, edge attr is just distance
        "batch": cat_batch,
        "y": cat_targets,
        "id": all_ids,
    }
