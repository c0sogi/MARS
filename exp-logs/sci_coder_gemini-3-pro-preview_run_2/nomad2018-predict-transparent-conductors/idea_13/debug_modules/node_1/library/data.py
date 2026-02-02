import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import TargetScaler


class CrystalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def process_graph(row, input_dir, cutoff):
    """
    Reads an XYZ file and converts it to a PyG Data object.
    """
    file_path = os.path.join(input_dir, row["file_path"])

    # Read structure
    atoms = read(file_path)

    # Get atomic numbers (Z) as node features
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Compute neighbors with PBC
    # i: index of center atom, j: index of neighbor, d: distance
    # self_interaction=False avoids self-loops
    i, j, d = neighbor_list("ijd", atoms, cutoff, self_interaction=False)

    # Construct edge index
    edge_index = torch.stack(
        [torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)], dim=0
    )

    # Edge features: distances
    edge_attr = torch.tensor(d, dtype=torch.float).unsqueeze(1)

    # Targets
    if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
        y = torch.tensor(
            [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]],
            dtype=torch.float,
        ).unsqueeze(0)
    else:
        # Placeholder for test set
        y = torch.zeros((1, 2), dtype=torch.float)

    # Create Data object
    # x needs to be [num_nodes, 1] or [num_nodes] depending on embedding layer expectation.
    # Usually Embedding takes [num_nodes], but PyG often expects [num_nodes, num_features].
    # We will keep it as [num_nodes] for nn.Embedding.
    data = Data(x=atomic_numbers, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Store ID for reference
    data.id = torch.tensor([row["id"]], dtype=torch.long)

    return data


def save_graphs_to_npz(data_list, path):
    """
    Collates a list of Data objects into numpy arrays and saves to .npz
    This avoids pickle.
    """
    # Initialize lists to hold concatenated data
    all_x = []
    all_edge_index_0 = []
    all_edge_index_1 = []
    all_edge_attr = []
    all_y = []
    all_ids = []

    # Pointers to reconstruct the list
    # node_ptr: start index of nodes for each graph
    # edge_ptr: start index of edges for each graph
    node_ptr = [0]
    edge_ptr = [0]

    for data in data_list:
        all_x.append(data.x.numpy())
        all_edge_index_0.append(data.edge_index[0].numpy())
        all_edge_index_1.append(data.edge_index[1].numpy())
        all_edge_attr.append(data.edge_attr.numpy())
        all_y.append(data.y.numpy())
        all_ids.append(data.id.numpy())

        node_ptr.append(node_ptr[-1] + data.x.shape[0])
        edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

    # Concatenate
    if len(all_x) > 0:
        np_x = np.concatenate(all_x)
        np_edge_index = np.stack(
            [np.concatenate(all_edge_index_0), np.concatenate(all_edge_index_1)]
        )
        np_edge_attr = np.concatenate(all_edge_attr)
        np_y = np.concatenate(all_y)
        np_ids = np.concatenate(all_ids)
    else:
        # Handle empty case
        np_x = np.array([])
        np_edge_index = np.array([[], []])
        np_edge_attr = np.array([])
        np_y = np.array([])
        np_ids = np.array([])

    np_node_ptr = np.array(node_ptr)
    np_edge_ptr = np.array(edge_ptr)

    # Save
    np.savez_compressed(
        path,
        x=np_x,
        edge_index=np_edge_index,
        edge_attr=np_edge_attr,
        y=np_y,
        ids=np_ids,
        node_ptr=np_node_ptr,
        edge_ptr=np_edge_ptr,
    )
    print(f"Saved cache to {path}")


def load_graphs_from_npz(path):
    """
    Loads graph data from .npz and reconstructs list of Data objects.
    """
    print(f"Loading cache from {path}...")
    data = np.load(path)

    x = torch.from_numpy(data["x"])
    edge_index = torch.from_numpy(data["edge_index"])
    edge_attr = torch.from_numpy(data["edge_attr"])
    y = torch.from_numpy(data["y"])
    ids = torch.from_numpy(data["ids"])
    node_ptr = data["node_ptr"]
    edge_ptr = data["edge_ptr"]

    data_list = []
    num_graphs = len(node_ptr) - 1

    for i in range(num_graphs):
        # Slice nodes
        n_start, n_end = node_ptr[i], node_ptr[i + 1]
        g_x = x[n_start:n_end]

        # Slice edges
        e_start, e_end = edge_ptr[i], edge_ptr[i + 1]
        g_edge_index = edge_index[:, e_start:e_end]
        g_edge_attr = edge_attr[e_start:e_end]

        # Slice targets and ids
        g_y = y[i].unsqueeze(0)
        g_id = ids[i].unsqueeze(0)

        d = Data(x=g_x, edge_index=g_edge_index, edge_attr=g_edge_attr, y=g_y)
        d.id = g_id
        data_list.append(d)

    return data_list


def process_dataset(
    metadata_path, cache_path, input_dir, cutoff, load_cached_data=True
):
    """
    Orchestrates loading, processing, and caching of data.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            return load_graphs_from_npz(cache_path)
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Process graphs
    data_list = []
    for _, row in df.iterrows():
        data = process_graph(row, input_dir, cutoff)
        data_list.append(data)

    # Save to cache
    save_graphs_to_npz(data_list, cache_path)

    return data_list


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Main entry point to get DataLoaders.
    Handles scaling of training targets.
    """
    # Define cache paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_graphs.npz")

    # Process or load datasets
    print("Processing/Loading Training Data...")
    train_list = process_dataset(
        Config.TRAIN_METADATA_PATH,
        train_cache,
        Config.INPUT_DIR,
        Config.CUTOFF_RADIUS,
        load_cached_data,
    )

    print("Processing/Loading Validation Data...")
    val_list = process_dataset(
        Config.VAL_METADATA_PATH,
        val_cache,
        Config.INPUT_DIR,
        Config.CUTOFF_RADIUS,
        load_cached_data,
    )

    print("Processing/Loading Test Data...")
    test_list = process_dataset(
        Config.TEST_METADATA_PATH,
        test_cache,
        Config.INPUT_DIR,
        Config.CUTOFF_RADIUS,
        load_cached_data,
    )

    # Fit scaler on training targets
    print("Fitting Target Scaler...")
    # Extract all y values from train_list into a single tensor
    train_y = torch.cat([d.y for d in train_list], dim=0)

    scaler = TargetScaler(device=Config.DEVICE)
    scaler.fit(train_y)

    # Save scaler for inference later
    scaler_path = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
    scaler.save(scaler_path)
    print(f"Scaler saved to {scaler_path}")
    print(f"Scaler Mean: {scaler.mean}, Std: {scaler.std}")

    # Transform targets in datasets (in-place modification of Data objects)
    # Note: We modify the Data objects in memory.
    for data in train_list:
        data.y = scaler.transform(data.y).cpu()  # Keep data on CPU for DataLoader

    for data in val_list:
        data.y = scaler.transform(data.y).cpu()

    # Test set targets are placeholders, no need to transform, but we will need scaler for inverse transform later.

    # Create Datasets
    train_dataset = CrystalGraphDataset(train_list)
    val_dataset = CrystalGraphDataset(val_list)
    test_dataset = CrystalGraphDataset(test_list)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler
