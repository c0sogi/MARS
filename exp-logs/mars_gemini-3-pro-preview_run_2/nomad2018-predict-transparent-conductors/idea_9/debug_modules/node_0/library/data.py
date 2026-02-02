import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase.io import read
from tqdm import tqdm

from library.config import Config
from library.utils import build_pbc_graph, StandardScaler


class CrystalDataset(Dataset):
    """
    Torch Geometric Dataset wrapper for crystal structures.
    """

    def __init__(self, data_list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def process_raw_data(metadata_path, mode="train"):
    """
    Reads metadata, loads geometry files, constructs graphs, and extracts global features.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        list: A list of torch_geometric.data.Data objects.
    """
    df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG mode: processing only {len(df)} samples for {mode}.")

    data_list = []

    # Iterate over the metadata to process each crystal
    # Using tqdm for progress tracking (though output is suppressed in final run usually)
    for _, row in df.iterrows():
        mat_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load atoms object using ASE
        atoms = read(full_path)

        # 1. Build Graph (Local Stream)
        # edge_src, edge_dst, edge_dist are tensors
        edge_src, edge_dst, edge_dist = build_pbc_graph(
            atoms, cutoff=Config.CUTOFF_RADIUS, max_neighbors=Config.MAX_NEIGHBORS
        )
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_attr = edge_dist  # Kept as distance; model handles RBF expansion

        # Node Features: Map chemical symbols to integers based on Config
        symbols = atoms.get_chemical_symbols()
        x = torch.tensor([Config.ATOM_MAP[s] for s in symbols], dtype=torch.long)

        # 2. Extract Global Features (Global Stream)
        # Lattice parameters: [a, b, c, alpha, beta, gamma]
        cell_par = atoms.cell.cellpar()

        # Composition: Fractions of O, Al, Ga, In
        # Config.ATOM_MAP = {"O": 0, "Al": 1, "Ga": 2, "In": 3}
        # We ensure the order matches the integer indices 0, 1, 2, 3
        counts = np.zeros(Config.NUM_ATOM_TYPES)
        for s in symbols:
            counts[Config.ATOM_MAP[s]] += 1
        fractions = counts / len(symbols)

        # Concatenate lattice and composition
        global_feat_np = np.concatenate([cell_par, fractions])
        global_feat = torch.tensor(global_feat_np, dtype=torch.float)

        # 3. Targets
        if mode in ["train", "val"]:
            y = torch.tensor([row[t] for t in Config.TARGET_COLS], dtype=torch.float)
        else:
            # Dummy target for test set to maintain consistent shape
            y = torch.zeros(len(Config.TARGET_COLS), dtype=torch.float)

        # Create Data object
        # We unsqueeze global_feat and y to add batch dimension [1, F]
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_feat=global_feat.unsqueeze(0),
            y=y.unsqueeze(0),
            id=torch.tensor([mat_id], dtype=torch.long),
        )
        data_list.append(data)

    return data_list


def save_to_cache(data_list, cache_path):
    """
    Saves a list of Data objects to an .npz file by flattening tensors.
    This avoids pickle and allows efficient storage.
    """
    # Arrays to hold concatenated data
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    global_feat_list = []
    y_list = []
    id_list = []

    # Pointer arrays to reconstruct the graph structure
    x_ptr = [0]
    edge_ptr = [0]

    for data in data_list:
        x_list.append(data.x.numpy())
        edge_index_list.append(data.edge_index.numpy())
        edge_attr_list.append(data.edge_attr.numpy())
        global_feat_list.append(data.global_feat.numpy())
        y_list.append(data.y.numpy())
        id_list.append(data.id.numpy())

        x_ptr.append(x_ptr[-1] + data.x.shape[0])
        edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    np.savez(
        cache_path,
        x=np.concatenate(x_list),
        x_ptr=np.array(x_ptr),
        edge_index=np.concatenate(edge_index_list, axis=1),
        edge_ptr=np.array(edge_ptr),
        edge_attr=np.concatenate(edge_attr_list),
        global_feat=np.concatenate(global_feat_list, axis=0),
        y=np.concatenate(y_list, axis=0),
        id=np.concatenate(id_list, axis=0),
    )
    print(f"Saved cached data to {cache_path}")


def load_from_cache(cache_path):
    """
    Loads data from .npz and reconstructs list of Data objects.
    """
    print(f"Loading cached data from {cache_path}...")
    data = np.load(cache_path)

    x_all = torch.from_numpy(data["x"])
    x_ptr = data["x_ptr"]
    edge_index_all = torch.from_numpy(data["edge_index"])
    edge_ptr = data["edge_ptr"]
    edge_attr_all = torch.from_numpy(data["edge_attr"])
    global_feat_all = torch.from_numpy(data["global_feat"])
    y_all = torch.from_numpy(data["y"])
    id_all = torch.from_numpy(data["id"])

    data_list = []
    num_samples = len(x_ptr) - 1

    for i in range(num_samples):
        # Slice node features
        x = x_all[x_ptr[i] : x_ptr[i + 1]]
        # Slice edges
        edge_index = edge_index_all[:, edge_ptr[i] : edge_ptr[i + 1]]
        edge_attr = edge_attr_all[edge_ptr[i] : edge_ptr[i + 1]]
        # Slice globals and targets (already batched in storage)
        global_feat = global_feat_all[i].unsqueeze(0)
        y = y_all[i].unsqueeze(0)
        mat_id = id_all[i].unsqueeze(0)

        d = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_feat=global_feat,
            y=y,
            id=mat_id,
        )
        data_list.append(d)

    return data_list


def get_dataset(metadata_path, cache_path, mode, load_cached_data=True):
    """
    Retrieves the dataset, either from cache or by processing raw files.
    """
    if load_cached_data and os.path.exists(cache_path):
        try:
            return load_from_cache(cache_path)
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    data_list = process_raw_data(metadata_path, mode)
    save_to_cache(data_list, cache_path)
    return data_list


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders for the pipeline.
    1. Loads train/val/test datasets.
    2. Fits StandardScalers on training data (globals and targets).
    3. Transforms all datasets using these scalers.
    4. Returns DataLoaders and the target scaler (for inverse transform later).
    """
    # 1. Load Datasets
    train_list = get_dataset(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_GRAPHS_CACHE, "train", load_cached_data
    )
    val_list = get_dataset(
        Config.VAL_METADATA_PATH, Config.VAL_GRAPHS_CACHE, "val", load_cached_data
    )
    test_list = get_dataset(
        Config.TEST_METADATA_PATH, Config.TEST_GRAPHS_CACHE, "test", load_cached_data
    )

    # 2. Initialize and Fit Scalers
    # We use CPU for scaler fitting to avoid GPU memory overhead during data prep
    global_scaler = StandardScaler(device="cpu")
    target_scaler = StandardScaler(device="cpu")

    print("Fitting scalers on training data...")
    # Collect all training global features and targets
    train_globals = torch.cat([d.global_feat for d in train_list], dim=0)
    train_targets = torch.cat([d.y for d in train_list], dim=0)

    global_scaler.fit(train_globals)
    target_scaler.fit(train_targets)

    # Save scalers for inference/reproducibility
    global_scaler.save(Config.SCALER_CACHE.replace(".npz", "_global.npz"))
    target_scaler.save(Config.SCALER_CACHE.replace(".npz", "_target.npz"))

    # 3. Apply Scaling
    # We modify the Data objects in-place.

    def apply_scaling(d_list, scale_y=True):
        for data in d_list:
            # Scale inputs
            data.global_feat = global_scaler.transform(data.global_feat)
            # Scale targets if required (Train/Val)
            if scale_y:
                data.y = target_scaler.transform(data.y)

    apply_scaling(train_list, scale_y=True)
    apply_scaling(val_list, scale_y=True)
    # For test set, we only scale inputs. Targets are dummy zeros.
    apply_scaling(test_list, scale_y=False)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        CrystalDataset(train_list),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        CrystalDataset(val_list),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        CrystalDataset(test_list),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, target_scaler
