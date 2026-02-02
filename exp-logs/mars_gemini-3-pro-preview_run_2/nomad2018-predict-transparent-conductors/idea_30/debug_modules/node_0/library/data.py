import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import ase.io
from ase.neighborlist import neighbor_list
from library.config import Config

# Initialize Config
config = Config()


def process_structure(file_path, targets=None):
    """
    Reads an XYZ file and converts it into a PyTorch Geometric Data object.

    Args:
        file_path (str): Relative path to the geometry.xyz file.
        targets (list or np.array, optional): Target values [formation_energy, bandgap].

    Returns:
        torch_geometric.data.Data: Graph representation of the crystal.
    """
    full_path = os.path.join(config.input_dir, file_path)

    # Read structure using ASE
    try:
        atoms = ase.io.read(full_path)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None

    # Node features: Atomic numbers (Z)
    # We use atomic numbers directly. Embedding layer in model will handle mapping.
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Edge construction: PBC-aware neighbor search
    # neighbor_list returns (i, j, d, D) where i, j are indices, d is distance, D is vector
    # We only need i, j, d.
    # cutoff is defined in Config
    cutoff = config.cutoff_radius
    i, j, d = neighbor_list("ijd", atoms, cutoff)

    # Create edge_index and edge_attr
    edge_index = torch.stack(
        [torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)], dim=0
    )

    # Edge features: Distances (will be expanded by RBF in the model)
    edge_attr = torch.tensor(d, dtype=torch.float).unsqueeze(1)

    # Targets
    y = None
    if targets is not None:
        y = torch.tensor(targets, dtype=torch.float).view(1, -1)

    # Create Data object
    data = Data(x=atomic_numbers, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Store number of nodes for batching
    data.num_nodes = len(atomic_numbers)

    return data


def save_graphs(graphs, path):
    """
    Saves a list of PyG Data objects to a compressed .npz file.
    This avoids pickle for the heavy data arrays.
    """
    all_x = []
    all_edge_index_src = []
    all_edge_index_dst = []
    all_edge_attr = []
    all_y = []

    node_ptr = [0]
    edge_ptr = [0]

    for data in graphs:
        all_x.append(data.x.numpy())
        all_edge_index_src.append(data.edge_index[0].numpy())
        all_edge_index_dst.append(data.edge_index[1].numpy())
        all_edge_attr.append(data.edge_attr.numpy())

        # Handle targets (might be None for test set, store as NaN)
        if data.y is not None:
            all_y.append(data.y.numpy())
        else:
            all_y.append(np.full((1, 2), np.nan))

        node_ptr.append(node_ptr[-1] + data.num_nodes)
        edge_ptr.append(edge_ptr[-1] + data.num_edges)

    # Concatenate all arrays
    # Check if we have any data to save
    if not all_x:
        print("Warning: No graphs to save.")
        return

    x_np = np.concatenate(all_x)
    edge_src_np = np.concatenate(all_edge_index_src)
    edge_dst_np = np.concatenate(all_edge_index_dst)
    edge_attr_np = np.concatenate(all_edge_attr)
    y_np = np.concatenate(all_y)
    node_ptr_np = np.array(node_ptr, dtype=np.int64)
    edge_ptr_np = np.array(edge_ptr, dtype=np.int64)

    np.savez_compressed(
        path,
        x=x_np,
        edge_src=edge_src_np,
        edge_dst=edge_dst_np,
        edge_attr=edge_attr_np,
        y=y_np,
        node_ptr=node_ptr_np,
        edge_ptr=edge_ptr_np,
    )
    print(f"Saved {len(graphs)} graphs to {path}")


def load_graphs(path):
    """
    Loads graphs from a .npz file created by save_graphs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")

    print(f"Loading graphs from {path}...")
    data = np.load(path)

    x = data["x"]
    edge_src = data["edge_src"]
    edge_dst = data["edge_dst"]
    edge_attr = data["edge_attr"]
    y = data["y"]
    node_ptr = data["node_ptr"]
    edge_ptr = data["edge_ptr"]

    graphs = []
    num_graphs = len(node_ptr) - 1

    for i in range(num_graphs):
        n_start, n_end = node_ptr[i], node_ptr[i + 1]
        e_start, e_end = edge_ptr[i], edge_ptr[i + 1]

        # Reconstruct tensors
        g_x = torch.from_numpy(x[n_start:n_end]).long()

        # Edge indices
        src = torch.from_numpy(edge_src[e_start:e_end]).long()
        dst = torch.from_numpy(edge_dst[e_start:e_end]).long()
        g_edge_index = torch.stack([src, dst], dim=0)

        g_edge_attr = torch.from_numpy(edge_attr[e_start:e_end]).float()

        # Targets
        g_y_np = y[i : i + 1]
        if np.isnan(g_y_np).any():
            g_y = None
        else:
            g_y = torch.from_numpy(g_y_np).float()

        data_obj = Data(x=g_x, edge_index=g_edge_index, edge_attr=g_edge_attr, y=g_y)
        data_obj.num_nodes = n_end - n_start
        graphs.append(data_obj)

    print(f"Loaded {len(graphs)} graphs.")
    return graphs


class CrystalGraphDataset(Dataset):
    """
    Custom PyG Dataset wrapper.
    """

    def __init__(self, data_list, root=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_dataloaders(load_cached_data=True, batch_size=None):
    """
    Main function to prepare dataloaders.

    Args:
        load_cached_data (bool): If True, attempts to load processed graphs from cache.
        batch_size (int): Batch size override. If None, uses Config.batch_size.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.batch_size

    # Cache paths
    train_cache = os.path.join(config.cache_dir, "train_graphs.npz")
    val_cache = os.path.join(config.cache_dir, "val_graphs.npz")
    test_cache = os.path.join(config.cache_dir, "test_graphs.npz")

    train_graphs = []
    val_graphs = []
    test_graphs = []

    # --- Train Data ---
    if load_cached_data and os.path.exists(train_cache):
        train_graphs = load_graphs(train_cache)
    else:
        print("Processing Train Data...")
        df_train = pd.read_csv(config.train_metadata_path)
        for _, row in df_train.iterrows():
            targets = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            graph = process_structure(row["file_path"], targets)
            if graph:
                train_graphs.append(graph)
        save_graphs(train_graphs, train_cache)

    # --- Val Data ---
    if load_cached_data and os.path.exists(val_cache):
        val_graphs = load_graphs(val_cache)
    else:
        print("Processing Validation Data...")
        df_val = pd.read_csv(config.val_metadata_path)
        for _, row in df_val.iterrows():
            targets = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            graph = process_structure(row["file_path"], targets)
            if graph:
                val_graphs.append(graph)
        save_graphs(val_graphs, val_cache)

    # --- Test Data ---
    if load_cached_data and os.path.exists(test_cache):
        test_graphs = load_graphs(test_cache)
    else:
        print("Processing Test Data...")
        df_test = pd.read_csv(config.test_metadata_path)
        for _, row in df_test.iterrows():
            # No targets for test
            graph = process_structure(row["file_path"], targets=None)
            # We attach the ID to the graph object for submission mapping
            if graph:
                graph.id = row["id"]
                test_graphs.append(graph)
        save_graphs(test_graphs, test_cache)

    # Create Datasets
    train_dataset = CrystalGraphDataset(train_graphs)
    val_dataset = CrystalGraphDataset(val_graphs)
    test_dataset = CrystalGraphDataset(test_graphs)

    # Create Loaders
    # Use num_workers=0 for safety in some environments, or >0 for speed
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader
