import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, Batch
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    CUTOFF_RADIUS,
    TARGET_COLS,
    DEBUG_MODE,
    DEBUG_DATA_SIZE,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)


def save_graphs_to_npz(data_list, path):
    """
    Saves a list of PyG Data objects to a compressed .npz file without using pickle.
    """
    # Use PyG Batch to collate everything efficiently
    batch = Batch.from_data_list(data_list)

    # Extract components
    # x: (N, 1) atomic numbers
    x = batch.x.numpy().astype(np.int32)

    # edge_index: (2, E)
    edge_index = batch.edge_index.numpy().astype(np.int32)

    # edge_attr: (E, 1) distances
    edge_attr = batch.edge_attr.numpy().astype(np.float32)

    # y: (G, 2) targets (formation, bandgap)
    # Handle case where y might be None (test set)
    if batch.y is not None:
        y = batch.y.numpy().astype(np.float32)
    else:
        y = np.array([], dtype=np.float32)

    # ptr: (G+1, ) node pointers to slice x
    ptr = batch.ptr.numpy().astype(np.int32)

    # Save
    np.savez_compressed(
        path, x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, ptr=ptr
    )


def load_graphs_from_npz(path):
    """
    Loads graphs from a .npz file and returns a list of PyG Data objects.
    """
    data = np.load(path)

    x_all = torch.from_numpy(data["x"]).long()
    edge_index_all = torch.from_numpy(data["edge_index"]).long()
    edge_attr_all = torch.from_numpy(data["edge_attr"]).float()
    ptr = torch.from_numpy(data["ptr"]).long()

    y_all = None
    if "y" in data and data["y"].size > 0:
        y_all = torch.from_numpy(data["y"]).float()

    data_list = []
    num_graphs = len(ptr) - 1

    for i in range(num_graphs):
        # Node indices for this graph
        start_node = ptr[i]
        end_node = ptr[i + 1]

        # Slice node features
        x = x_all[start_node:end_node]

        # Identify edges belonging to this graph
        # Edges are global indices in the batch, we need to filter those connecting nodes in this graph
        mask = (edge_index_all[0] >= start_node) & (edge_index_all[0] < end_node)
        edge_index = edge_index_all[:, mask]
        edge_attr = edge_attr_all[mask]

        # Shift edge indices back to local 0-based indexing
        edge_index = edge_index - start_node

        # Targets
        y = None
        if y_all is not None:
            y = y_all[i].unsqueeze(0)  # (1, 2)

        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data_obj)

    return data_list


def process_geometry(file_path, cutoff):
    """
    Reads an XYZ file and constructs graph features.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    # Read atoms
    # ASE reads lattice vectors from the file automatically
    # The files have .xyz extension but contain FHI-aims formatted data
    atoms = read(full_path, format="aims")
    atoms.pbc = True  # Ensure PBC is enabled

    # Atomic numbers as node features
    # shape: (num_nodes, 1)
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).unsqueeze(1)

    # Neighbor search
    # i: source, j: target, d: distance
    # Using neighbor_list from ase
    # self_interaction=False to avoid loops
    i_idx, j_idx, d_vals = neighbor_list("ijd", atoms, cutoff, self_interaction=False)

    edge_index = torch.stack(
        [torch.tensor(i_idx, dtype=torch.long), torch.tensor(j_idx, dtype=torch.long)],
        dim=0,
    )

    edge_attr = torch.tensor(d_vals, dtype=torch.float).unsqueeze(1)

    return z, edge_index, edge_attr


class CrystalGraphDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_path,
        load_cached_data=True,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            metadata_path: Path to the metadata CSV (train/val/test).
            cache_path: Path to the .npz cache file.
            load_cached_data: Whether to try loading from cache.
        """
        super().__init__(None, transform, pre_transform)
        self.metadata_path = metadata_path
        self.cache_path = cache_path
        self.load_cached_data = load_cached_data
        self.data_list = []

        self._process()

    @property
    def raw_file_names(self):
        return [self.metadata_path]

    @property
    def processed_file_names(self):
        return [self.cache_path]

    def _process(self):
        # Logic Flow as requested
        loaded = False

        # 1. Try to load
        if self.load_cached_data and os.path.exists(self.cache_path):
            try:
                self.data_list = load_graphs_from_npz(self.cache_path)
                loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                loaded = False

        # 2. If failed or forced to recompute
        if not loaded:
            df = pd.read_csv(self.metadata_path)

            # Debug mode
            if DEBUG_MODE:
                df = df.iloc[:DEBUG_DATA_SIZE]

            data_list = []

            for idx, row in df.iterrows():
                # Geometry processing
                z, edge_index, edge_attr = process_geometry(
                    row["file_path"], CUTOFF_RADIUS
                )

                # Targets
                y = None
                # Check if target columns exist in metadata
                if all(col in row for col in TARGET_COLS):
                    targets = row[TARGET_COLS].values.astype(np.float32)
                    y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)  # (1, 2)

                data = Data(x=z, edge_index=edge_index, edge_attr=edge_attr, y=y)
                data_list.append(data)

            self.data_list = data_list

            # Save to cache
            save_graphs_to_npz(self.data_list, self.cache_path)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_train_val_datasets(load_cached=True):
    """
    Helper to get train and val datasets.
    """
    train_cache = os.path.join(CACHE_DIR, "train_graphs.npz")
    val_cache = os.path.join(CACHE_DIR, "val_graphs.npz")

    train_dataset = CrystalGraphDataset(
        TRAIN_METADATA_PATH, train_cache, load_cached_data=load_cached
    )
    val_dataset = CrystalGraphDataset(
        VAL_METADATA_PATH, val_cache, load_cached_data=load_cached
    )

    return train_dataset, val_dataset


def get_test_dataset(load_cached=True):
    """
    Helper to get test dataset.
    """
    test_cache = os.path.join(CACHE_DIR, "test_graphs.npz")
    test_dataset = CrystalGraphDataset(
        TEST_METADATA_PATH, test_cache, load_cached_data=load_cached
    )

    return test_dataset
