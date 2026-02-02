import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config


class CrystalDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def process_structure(file_path, targets=None):
    """
    Reads an XYZ file and converts it into a PyTorch Geometric Data object.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    try:
        # Cite debug_lesson_1: Explicitly specify format='aims' because the file extension .xyz is misleading
        # and the content is actually in FHI-aims format.
        atoms = read(full_path, format="aims")
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None

    # Node features: Atomic numbers
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Edge features: Distances within cutoff
    # 'i' is source, 'j' is target (or vice versa, symmetric for undirected usually, but neighbor_list returns directed pairs)
    # We want directed edges for GNN message passing (j -> i)
    i, j, d = neighbor_list("ijd", atoms, Config.CUTOFF_RADIUS, self_interaction=False)

    edge_index = torch.stack(
        [torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)], dim=0
    )
    edge_attr = torch.tensor(d, dtype=torch.float)

    # Targets
    y = None
    if targets is not None:
        y = torch.tensor(targets, dtype=torch.float).view(1, -1)

    # Create Data object
    # We store 'z' as 'x' for convention, or keep as 'z'.
    # The model description says "embedded based on atomic number", so 'z' is appropriate.
    data = Data(z=z, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=len(z))

    return data


def save_graphs(graphs, path):
    """
    Saves a list of PyTorch Geometric Data objects to a compressed .npz file.
    We flatten the data for efficient storage.
    """
    # Arrays to store
    all_z = []
    all_edge_src = []
    all_edge_dst = []
    all_edge_attr = []
    all_y = []

    node_ptr = [0]
    edge_ptr = [0]

    for data in graphs:
        all_z.append(data.z.numpy())
        all_edge_src.append(data.edge_index[0].numpy())
        all_edge_dst.append(data.edge_index[1].numpy())
        all_edge_attr.append(data.edge_attr.numpy())

        if data.y is not None:
            all_y.append(data.y.numpy())
        else:
            # Placeholder for test set if needed, though usually None is fine.
            # But npz expects arrays. Let's store NaNs for missing targets.
            all_y.append(np.full((1, 2), np.nan))

        node_ptr.append(node_ptr[-1] + data.num_nodes)
        edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

    # Concatenate
    if not all_z:
        return  # Empty list

    np.savez_compressed(
        path,
        z=np.concatenate(all_z),
        edge_src=np.concatenate(all_edge_src),
        edge_dst=np.concatenate(all_edge_dst),
        edge_attr=np.concatenate(all_edge_attr),
        y=np.concatenate(all_y),
        node_ptr=np.array(node_ptr, dtype=np.int64),
        edge_ptr=np.array(edge_ptr, dtype=np.int64),
    )
    print(f"Saved {len(graphs)} graphs to {path}")


def load_graphs(path):
    """
    Loads graphs from a .npz file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")

    print(f"Loading cached graphs from {path}...")
    data = np.load(path)

    z = torch.from_numpy(data["z"])
    edge_src = torch.from_numpy(data["edge_src"])
    edge_dst = torch.from_numpy(data["edge_dst"])
    edge_attr = torch.from_numpy(data["edge_attr"])
    y = torch.from_numpy(data["y"])
    node_ptr = data["node_ptr"]
    edge_ptr = data["edge_ptr"]

    graphs = []
    num_graphs = len(node_ptr) - 1

    for i in range(num_graphs):
        # Slice nodes
        n_start, n_end = node_ptr[i], node_ptr[i + 1]
        g_z = z[n_start:n_end]

        # Slice edges
        e_start, e_end = edge_ptr[i], edge_ptr[i + 1]
        g_src = edge_src[e_start:e_end]
        g_dst = edge_dst[e_start:e_end]
        g_edge_index = torch.stack([g_src, g_dst], dim=0)
        g_edge_attr = edge_attr[e_start:e_end]

        # Slice targets
        g_y = y[i : i + 1]
        if torch.isnan(g_y).any():
            g_y = None

        graphs.append(
            Data(
                z=g_z,
                edge_index=g_edge_index,
                edge_attr=g_edge_attr,
                y=g_y,
                num_nodes=n_end - n_start,
            )
        )

    print(f"Loaded {len(graphs)} graphs.")
    return graphs


def get_dataset(metadata_path, cache_path, load_cached=True, dataset_size=None):
    """
    Generic function to load or process a dataset.
    """
    # 1. Try loading from cache
    if load_cached and os.path.exists(cache_path):
        try:
            graphs = load_graphs(cache_path)
            if dataset_size:
                graphs = graphs[:dataset_size]
            return CrystalDataset(graphs)
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if dataset_size:
        df = df.iloc[:dataset_size]

    graphs = []
    for _, row in df.iterrows():
        # Extract targets if available
        targets = None
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            targets = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]

        graph = process_structure(row["file_path"], targets)
        if graph is not None:
            graphs.append(graph)

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_graphs(graphs, cache_path)

    return CrystalDataset(graphs)


def get_dataloaders(load_cached_data=True, dataset_size=None):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to try loading from .npz cache.
        dataset_size (int, optional): Limit size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Train
    train_dataset = get_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_GRAPHS_PATH,
        load_cached=load_cached_data,
        dataset_size=dataset_size,
    )

    # Val
    val_dataset = get_dataset(
        Config.VAL_METADATA_PATH,
        Config.VAL_GRAPHS_PATH,
        load_cached=load_cached_data,
        dataset_size=dataset_size,
    )

    # Test
    test_dataset = get_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_GRAPHS_PATH,
        load_cached=load_cached_data,
        dataset_size=dataset_size,
    )

    # Create DataLoaders
    # PyG DataLoader handles batching of graphs (diagonal stacking of adjacency matrices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
