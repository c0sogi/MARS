import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import StandardScaler


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Calculate width based on the spacing between centers
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def get_pbc_graph(file_path):
    """
    Constructs a PyG Data object from an XYZ file using PBC-aware neighbor search.
    """
    # Load atom structure
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    atoms = read(full_path, format="aims")

    # Node features: Atomic numbers (Z)
    # We keep them as integers/longs for embedding lookup
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Edge construction with PBC
    # i: source indices, j: target indices, d: distances
    cut = Config.CUTOFF_RADIUS
    i, j, d = neighbor_list("ijD", atoms, cut)

    # Convert to tensors
    edge_index = torch.stack(
        [torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)], dim=0
    )
    distances = torch.tensor(d, dtype=torch.float)

    # Edge Features: Gaussian RBF Expansion
    # We instantiate the smearing function here to pre-compute features
    # In a full pipeline, this might be a global instance, but here we keep it local or static
    rbf_expander = GaussianSmearing(
        start=Config.RBF_MIN, stop=Config.RBF_MAX, num_gaussians=Config.NUM_RBF_BINS
    )
    edge_attr = rbf_expander(distances)

    # Create Data object
    # Note: y (targets) will be added later by the Dataset class
    data = Data(
        x=atomic_numbers,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(atoms),
    )

    return data


def save_graphs_to_npz(data_list, path):
    """
    Saves a list of PyG Data objects to a compressed NPZ file using numpy arrays.
    Avoids pickle.
    """
    # Collect all data into lists
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_y = []
    n_nodes = []
    n_edges = []

    for data in data_list:
        all_x.append(data.x.numpy())
        all_edge_index.append(data.edge_index.numpy())
        all_edge_attr.append(data.edge_attr.numpy())

        # Handle targets if they exist
        if hasattr(data, "y") and data.y is not None:
            all_y.append(data.y.numpy())
        else:
            # Placeholder for test set if y is missing
            all_y.append(np.array([np.nan, np.nan]))

        n_nodes.append(data.num_nodes)
        n_edges.append(data.num_edges)

    # Concatenate
    # Note: edge_index is (2, E), we concatenate along E (axis 1)
    cat_x = np.concatenate(all_x, axis=0)
    cat_edge_index = np.concatenate(all_edge_index, axis=1)
    cat_edge_attr = np.concatenate(all_edge_attr, axis=0)
    cat_y = np.stack(all_y, axis=0)
    arr_n_nodes = np.array(n_nodes, dtype=np.int64)
    arr_n_edges = np.array(n_edges, dtype=np.int64)

    # Save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x=cat_x,
        edge_index=cat_edge_index,
        edge_attr=cat_edge_attr,
        y=cat_y,
        n_nodes=arr_n_nodes,
        n_edges=arr_n_edges,
    )


def load_graphs_from_npz(path):
    """
    Loads a list of PyG Data objects from an NPZ file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")

    data_dict = np.load(path)

    cat_x = data_dict["x"]
    cat_edge_index = data_dict["edge_index"]
    cat_edge_attr = data_dict["edge_attr"]
    cat_y = data_dict["y"]
    n_nodes = data_dict["n_nodes"]
    n_edges = data_dict["n_edges"]

    data_list = []

    # Pointers for slicing
    ptr_x = 0
    ptr_edge = 0

    for i in range(len(n_nodes)):
        num_n = n_nodes[i]
        num_e = n_edges[i]

        # Slice
        x = torch.from_numpy(cat_x[ptr_x : ptr_x + num_n]).long()
        # edge_index was concatenated along axis 1
        edge_index = torch.from_numpy(
            cat_edge_index[:, ptr_edge : ptr_edge + num_e]
        ).long()
        edge_attr = torch.from_numpy(cat_edge_attr[ptr_edge : ptr_edge + num_e]).float()

        y_val = cat_y[i]
        if np.isnan(y_val).any():
            y = None
        else:
            y = torch.from_numpy(y_val).float().view(1, -1)

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_n
        )
        data_list.append(data)

        ptr_x += num_n
        ptr_edge += num_e

    return data_list


class CrystalDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_path,
        mode="train",
        load_cached_data=True,
        scaler=None,
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.cache_path = cache_path
        self.mode = mode
        self.data_list = []

        # Load Metadata
        self.df = pd.read_csv(metadata_path)

        # Caching Logic
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} graphs from {cache_path}...")
            self.data_list = load_graphs_from_npz(cache_path)
        else:
            print(f"Processing {mode} graphs from scratch...")
            self.process_data()
            print(f"Saving {mode} graphs to {cache_path}...")
            save_graphs_to_npz(self.data_list, cache_path)

        # Handle Scaling
        # If training, we fit the scaler if not provided, or use provided one
        # If val/test, we assume scaler is provided (fitted on train)
        if self.mode == "train":
            if scaler is None:
                self.scaler = StandardScaler()
                # Extract all y values
                all_y = torch.cat([d.y for d in self.data_list], dim=0)
                self.scaler.fit(all_y)
                # Save scaler for inference
                self.scaler.save(Config.TARGET_SCALER_PATH)
            else:
                self.scaler = scaler
        else:
            self.scaler = scaler

        # Apply scaling to y in memory
        if self.scaler is not None:
            for data in self.data_list:
                # Use getattr to safely check for y, as it might be missing in test data
                y = getattr(data, "y", None)
                if y is not None:
                    # Transform and move back to CPU for DataLoader pinning
                    data.y = self.scaler.transform(y).cpu()

    def process_data(self):
        for idx, row in self.df.iterrows():
            # Construct graph
            data = get_pbc_graph(row["file_path"])

            # Add target if available
            if self.mode in ["train", "val"]:
                targets = row[Config.TARGET_COLS].values.astype(np.float32)
                data.y = torch.tensor(targets).view(1, -1)

            # We can also add ID for tracking
            data.id = torch.tensor([row["id"]], dtype=torch.long)

            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_loaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Returns: train_loader, val_loader, test_loader, scaler
    """
    # 1. Train Dataset
    train_dataset = CrystalDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_GRAPHS_CACHE,
        mode="train",
        load_cached_data=load_cached_data,
        scaler=None,  # Will fit new scaler
    )

    # 2. Validation Dataset
    # Use scaler fitted on training data
    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.VAL_GRAPHS_CACHE,
        mode="val",
        load_cached_data=load_cached_data,
        scaler=train_dataset.scaler,
    )

    # 3. Test Dataset
    test_dataset = CrystalDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=Config.TEST_GRAPHS_CACHE,
        mode="test",
        load_cached_data=load_cached_data,
        scaler=train_dataset.scaler,  # Not strictly needed for y, but good for consistency
    )

    # Create Loaders
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

    return train_loader, val_loader, test_loader, train_dataset.scaler
