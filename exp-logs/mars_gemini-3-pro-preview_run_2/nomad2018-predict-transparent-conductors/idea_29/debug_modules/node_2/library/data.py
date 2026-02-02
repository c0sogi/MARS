import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library import config
from library import utils


def process_structure(file_path, cutoff=5.0):
    """
    Reads an XYZ file and converts it into a graph representation.

    Args:
        file_path (str): Path to the geometry.xyz file.
        cutoff (float): Cutoff radius for neighbor search.

    Returns:
        x (torch.Tensor): Node features (atomic numbers).
        edge_index (torch.Tensor): Graph connectivity.
        edge_attr (torch.Tensor): Edge features (distances).
    """
    full_path = os.path.join(config.INPUT_DIR, file_path)
    atoms = read(full_path, format="aims")

    # Node features: Atomic numbers
    # We use get_atomic_numbers() which returns an integer array
    atomic_numbers = atoms.get_atomic_numbers()
    x = torch.tensor(atomic_numbers, dtype=torch.long)

    # Neighbor search with PBC
    # 'i': atom index, 'j': neighbor index, 'd': distance
    i, j, d = neighbor_list("ijd", atoms, cutoff)

    # Construct edge_index [2, num_edges]
    edge_index = torch.tensor(np.vstack((i, j)), dtype=torch.long)

    # Edge features: Distances [num_edges, 1]
    edge_attr = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

    return x, edge_index, edge_attr


def save_cached_dataset(path, data_list, ids):
    """
    Saves a list of Data objects to a compressed NPZ file by flattening arrays.
    """
    # Flatten data
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_y = []

    # Pointers to reconstruct the graphs
    ptr_x = [0]
    ptr_edge = [0]

    for data in data_list:
        all_x.append(data.x.numpy())
        all_edge_index.append(data.edge_index.numpy())
        all_edge_attr.append(data.edge_attr.numpy())

        if hasattr(data, "y") and data.y is not None:
            all_y.append(data.y.numpy())
        else:
            # Placeholder for test set
            all_y.append(np.array([-1.0, -1.0]))

        ptr_x.append(ptr_x[-1] + data.x.shape[0])
        ptr_edge.append(ptr_edge[-1] + data.edge_index.shape[1])

    # Concatenate
    cat_x = np.concatenate(all_x)
    cat_edge_index = np.concatenate(all_edge_index, axis=1)
    cat_edge_attr = np.concatenate(all_edge_attr)
    cat_y = np.vstack(all_y)

    np.savez_compressed(
        path,
        x=cat_x,
        edge_index=cat_edge_index,
        edge_attr=cat_edge_attr,
        y=cat_y,
        ptr_x=np.array(ptr_x),
        ptr_edge=np.array(ptr_edge),
        ids=np.array(ids),
    )
    print(f"Saved dataset to {path}")


def load_cached_dataset(path):
    """
    Loads a list of Data objects from a compressed NPZ file.
    """
    print(f"Loading cached dataset from {path}...")
    data = np.load(path)

    cat_x = torch.from_numpy(data["x"])
    cat_edge_index = torch.from_numpy(data["edge_index"])
    cat_edge_attr = torch.from_numpy(data["edge_attr"])
    cat_y = torch.from_numpy(data["y"])
    ptr_x = data["ptr_x"]
    ptr_edge = data["ptr_edge"]
    ids = data["ids"]

    data_list = []
    for i in range(len(ids)):
        # Slice nodes
        x = cat_x[ptr_x[i] : ptr_x[i + 1]]

        # Slice edges
        edge_index = cat_edge_index[:, ptr_edge[i] : ptr_edge[i + 1]]
        edge_attr = cat_edge_attr[ptr_edge[i] : ptr_edge[i + 1]]

        # Slice target
        y = cat_y[i].unsqueeze(0)

        # Create Data object
        # Note: We don't need to re-index edge_index because it was concatenated along dim 1
        # but the indices themselves refer to node positions relative to the start of the batch?
        # No, standard PyG Data objects usually have 0-based indices for that graph.
        # neighbor_list returns 0-based indices relative to the atoms object.
        # So no offset adjustment is needed if we store them as is.

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=int(ids[i]))
        data_list.append(d)

    return data_list


class CrystalGraphDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def prepare_dataset(metadata_path, cache_path, load_cached_data=True, is_test=False):
    """
    Prepares the dataset by either loading from cache or processing from scratch.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data_list = load_cached_dataset(cache_path)
            return CrystalGraphDataset(data_list)
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing dataset from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    data_list = []
    ids = []

    for idx, row in df.iterrows():
        file_path = row["file_path"]
        material_id = row["id"]

        # Process geometry
        x, edge_index, edge_attr = process_structure(
            file_path, cutoff=config.CUTOFF_RADIUS
        )

        # Get targets if not test set
        if not is_test:
            y = torch.tensor(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]],
                dtype=torch.float32,
            ).unsqueeze(0)
        else:
            y = None

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=material_id
        )
        data_list.append(data)
        ids.append(material_id)

    # 3. Save to cache
    save_cached_dataset(cache_path, data_list, ids)

    return CrystalGraphDataset(data_list)


def get_dataloaders(load_cached_data=True, batch_size=config.BATCH_SIZE):
    """
    Creates DataLoaders for train, validation, and test sets.
    Also fits and returns a StandardScaler on the training targets.

    Args:
        load_cached_data (bool): Whether to try loading processed data from cache.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        train_loader, val_loader, test_loader, target_scaler
    """
    # Define cache paths
    train_cache = os.path.join(config.CACHE_DIR, "train_graphs.npz")
    val_cache = os.path.join(config.CACHE_DIR, "val_graphs.npz")
    test_cache = os.path.join(config.CACHE_DIR, "test_graphs.npz")

    # Prepare datasets
    train_dataset = prepare_dataset(
        config.TRAIN_METADATA_PATH, train_cache, load_cached_data, is_test=False
    )
    val_dataset = prepare_dataset(
        config.VAL_METADATA_PATH, val_cache, load_cached_data, is_test=False
    )
    test_dataset = prepare_dataset(
        config.TEST_METADATA_PATH, test_cache, load_cached_data, is_test=True
    )

    # Fit scaler on training targets
    print("Fitting target scaler on training data...")
    # Extract all y values from the training dataset
    # We can iterate efficiently since it's a list in memory
    all_train_y = torch.cat([data.y for data in train_dataset], dim=0)

    target_scaler = utils.StandardScaler()
    target_scaler.fit(all_train_y)

    # Save scaler for inference later if needed
    scaler_path = os.path.join(config.CACHE_DIR, "target_scaler.npz")
    target_scaler.save(scaler_path)

    # Create loaders
    # We use num_workers > 0 for parallel loading if possible
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader, target_scaler
