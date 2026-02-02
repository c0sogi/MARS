import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import StandardScaler


def collate_graphs(batch):
    """
    Collates a list of graph dictionaries into a single batch.
    """
    # Initialize lists
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_y = []
    batch_idx = []

    node_offset = 0

    for i, graph in enumerate(batch):
        num_nodes = graph["x"].shape[0]

        # Node features
        all_x.append(graph["x"])

        # Edge indices (shifted by current node offset)
        all_edge_index.append(graph["edge_index"] + node_offset)

        # Edge attributes
        all_edge_attr.append(graph["edge_attr"])

        # Targets (if available)
        if graph["y"] is not None:
            all_y.append(graph["y"])

        # Batch index (maps each node to its graph index in the batch)
        batch_idx.append(torch.full((num_nodes,), i, dtype=torch.long))

        node_offset += num_nodes

    # Concatenate all
    x = torch.cat(all_x, dim=0)
    edge_index = torch.cat(all_edge_index, dim=1)
    edge_attr = torch.cat(all_edge_attr, dim=0)
    batch_vec = torch.cat(batch_idx, dim=0)

    y = torch.stack(all_y, dim=0) if all_y else None

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": y,
        "batch": batch_vec,
    }


class AtomGraphDataset(Dataset):
    def __init__(self, data_dict, scaler=None, mode="train"):
        """
        Args:
            data_dict: Dictionary containing concatenated numpy arrays and pointers.
            scaler: StandardScaler instance for target normalization.
            mode: 'train', 'val', or 'test'.
        """
        self.x_all = torch.from_numpy(data_dict["x_all"]).long()
        self.edge_index_all = torch.from_numpy(data_dict["edge_index_all"]).long()
        self.edge_attr_all = torch.from_numpy(data_dict["edge_attr_all"]).float()

        if mode != "test":
            y_raw = data_dict["y_all"]
            if scaler:
                y_scaled = scaler.transform(y_raw)
                self.y_all = torch.from_numpy(y_scaled).float()
            else:
                self.y_all = torch.from_numpy(y_raw).float()
        else:
            self.y_all = None

        self.node_ptr = data_dict["node_ptr"]
        self.edge_ptr = data_dict["edge_ptr"]
        self.num_graphs = len(self.node_ptr) - 1
        self.mode = mode

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        # Slice node features
        node_start = self.node_ptr[idx]
        node_end = self.node_ptr[idx + 1]
        x = self.x_all[node_start:node_end]

        # Slice edge features
        edge_start = self.edge_ptr[idx]
        edge_end = self.edge_ptr[idx + 1]
        edge_index = self.edge_index_all[:, edge_start:edge_end]
        edge_attr = self.edge_attr_all[edge_start:edge_end]

        # Targets
        y = self.y_all[idx] if self.y_all is not None else None

        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y}


def process_graphs(metadata_path, cache_path, load_cached_data=True):
    """
    Reads metadata, loads geometry files, constructs graphs, and caches/loads them.
    Returns a dictionary of concatenated arrays and pointers.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached graphs from {cache_path}...")
        try:
            with np.load(cache_path) as data:
                return {
                    "x_all": data["x_all"],
                    "edge_index_all": data["edge_index_all"],
                    "edge_attr_all": data["edge_attr_all"],
                    "y_all": data["y_all"] if "y_all" in data else None,
                    "node_ptr": data["node_ptr"],
                    "edge_ptr": data["edge_ptr"],
                }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing graphs from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Lists to store graph data
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    y_list = []

    # Pointers
    node_ptr = [0]
    edge_ptr = [0]

    total_nodes = 0
    total_edges = 0

    for idx, row in df.iterrows():
        # Load geometry
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        atoms = read(file_path)

        # Node features: Atomic numbers
        atomic_numbers = atoms.get_atomic_numbers()
        x_list.append(atomic_numbers)

        # Edge features: Neighbor list with PBC
        # i: indices of central atoms, j: indices of neighbors, d: distances
        i_indices, j_indices, distances = neighbor_list("ijd", atoms, Config.CUTOFF)

        # Construct edge index (2, E)
        edge_index = np.vstack((i_indices, j_indices))
        edge_index_list.append(edge_index)

        # Edge attributes (distances) (E, 1)
        edge_attr_list.append(distances.reshape(-1, 1))

        # Targets
        if "formation_energy_ev_natom" in row:
            targets = row[Config.TARGET_COLS].values.astype(np.float32)
            y_list.append(targets)

        # Update pointers
        n_nodes = len(atomic_numbers)
        n_edges = len(distances)

        total_nodes += n_nodes
        total_edges += n_edges

        node_ptr.append(total_nodes)
        edge_ptr.append(total_edges)

    # Concatenate all data
    x_all = np.concatenate(x_list)
    edge_index_all = np.concatenate(edge_index_list, axis=1)
    edge_attr_all = np.concatenate(edge_attr_list, axis=0)

    y_all = np.stack(y_list) if y_list else np.empty((0, 2))

    node_ptr = np.array(node_ptr)
    edge_ptr = np.array(edge_ptr)

    data_dict = {
        "x_all": x_all,
        "edge_index_all": edge_index_all,
        "edge_attr_all": edge_attr_all,
        "y_all": y_all,
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
    }

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **data_dict)
    print(f"Saved processed graphs to {cache_path}")

    return data_dict


def get_loaders(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Orchestrates data loading, processing, scaling, and DataLoader creation.
    """
    # Process/Load Train Data
    train_data = process_graphs(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_GRAPHS_CACHE, load_cached_data
    )

    # Fit Scaler on Train Targets
    scaler = StandardScaler()
    scaler.fit(train_data["y_all"])

    # Save Scaler state for inference
    os.makedirs(os.path.dirname(Config.TARGET_SCALER_CACHE), exist_ok=True)
    np.savez(
        Config.TARGET_SCALER_CACHE, mean=scaler.mean.numpy(), std=scaler.std.numpy()
    )

    # Process/Load Val Data
    val_data = process_graphs(
        Config.VAL_METADATA_PATH, Config.VAL_GRAPHS_CACHE, load_cached_data
    )

    # Process/Load Test Data
    test_data = process_graphs(
        Config.TEST_METADATA_PATH, Config.TEST_GRAPHS_CACHE, load_cached_data
    )

    # Create Datasets
    train_dataset = AtomGraphDataset(train_data, scaler, mode="train")
    val_dataset = AtomGraphDataset(val_data, scaler, mode="val")
    test_dataset = AtomGraphDataset(test_data, scaler=None, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler
