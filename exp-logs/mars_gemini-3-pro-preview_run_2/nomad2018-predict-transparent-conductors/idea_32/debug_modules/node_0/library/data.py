import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ase.io import read
from library.config import Config
from library.features import compute_pbc_radius_graph


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for Crystal Structures.
    Reads XYZ files and converts them to PyG Data objects with PBC-aware graphs.
    Implements manual caching using npz to avoid pickle restrictions.
    """

    def __init__(self, metadata_path, split_name, load_cached_data=True):
        self.metadata_path = metadata_path
        self.split_name = split_name
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}_graphs.npz")

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Debugging: subsample if configured
        if Config.DEBUG_SAMPLE_SIZE is not None:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        self.data_list = self._load_or_process(load_cached_data)

    def _load_or_process(self, load_cached_data):
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(
                    f"Loading cached {self.split_name} data from {self.cache_path}..."
                )
                return self._load_cache()
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Processing {self.split_name} data from scratch...")
        data_list = self._process_raw_data()

        # 3. Save to cache
        try:
            self._save_cache(data_list)
            print(f"Saved {self.split_name} data to {self.cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return data_list

    def _process_raw_data(self):
        data_list = []

        for idx, row in self.df.iterrows():
            # Path to geometry file
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Read atoms
            atoms = read(full_path)

            # Compute graph
            graph_dict = compute_pbc_radius_graph(
                atoms, cutoff=Config.CUTOFF, max_neighbors=Config.MAX_NEIGHBORS
            )

            # Extract features
            x = graph_dict["atomic_numbers"]
            edge_index = graph_dict["edge_index"]
            edge_dist = graph_dict["edge_dist"]
            edge_vector = graph_dict["edge_vector"]
            pos = graph_dict["pos"]
            cell = graph_dict["cell"]

            # Targets
            if self.split_name in ["train", "val"]:
                y = torch.tensor(
                    row[Config.TARGET_COLS].values.astype(np.float32), dtype=torch.float
                ).view(1, -1)
            else:
                # Test set placeholder
                y = torch.zeros((1, len(Config.TARGET_COLS)), dtype=torch.float)

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_dist,  # We pass distance as edge_attr for now, model handles it
                edge_vector=edge_vector,
                y=y,
                pos=pos,
                cell=cell.view(1, 3, 3),
                id=row["id"],  # Keep track of ID
            )
            data_list.append(data)

        return data_list

    def _save_cache(self, data_list):
        # Collate data into numpy arrays for storage
        # We need to store:
        # - concatenated arrays for x, edge_index, edge_attr, edge_vector, y, pos, cell, id
        # - slices (start indices) to reconstruct individual graphs

        # Initialize lists
        xs, edge_indices, edge_attrs, edge_vectors, ys, poss, cells, ids = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # Slices
        # We need slices for node-level feats (x, pos) and edge-level feats (edge_index, edge_attr, edge_vector)
        # y, cell, id are graph-level (1 per graph)

        x_counts = []
        edge_counts = []

        for data in data_list:
            xs.append(data.x.numpy())
            edge_indices.append(data.edge_index.numpy())
            edge_attrs.append(data.edge_attr.numpy())
            edge_vectors.append(data.edge_vector.numpy())
            ys.append(data.y.numpy())
            poss.append(data.pos.numpy())
            cells.append(data.cell.numpy())
            ids.append(data.id)

            x_counts.append(data.x.shape[0])
            edge_counts.append(data.edge_index.shape[1])

        # Concatenate
        save_dict = {
            "x": np.concatenate(xs, axis=0),
            "edge_index": np.concatenate(edge_indices, axis=1),
            "edge_attr": np.concatenate(edge_attrs, axis=0),
            "edge_vector": np.concatenate(edge_vectors, axis=0),
            "y": np.concatenate(ys, axis=0),
            "pos": np.concatenate(poss, axis=0),
            "cell": np.concatenate(cells, axis=0),
            "id": np.array(ids),
            "x_slices": np.cumsum([0] + x_counts),
            "edge_slices": np.cumsum([0] + edge_counts),
        }

        np.savez(self.cache_path, **save_dict)

    def _load_cache(self):
        data = np.load(self.cache_path)

        x_all = torch.from_numpy(data["x"])
        edge_index_all = torch.from_numpy(data["edge_index"])
        edge_attr_all = torch.from_numpy(data["edge_attr"])
        edge_vector_all = torch.from_numpy(data["edge_vector"])
        y_all = torch.from_numpy(data["y"])
        pos_all = torch.from_numpy(data["pos"])
        cell_all = torch.from_numpy(data["cell"])
        id_all = data["id"]  # keep as numpy or list

        x_slices = data["x_slices"]
        edge_slices = data["edge_slices"]

        data_list = []
        num_graphs = len(id_all)

        for i in range(num_graphs):
            # Node slice
            n_start, n_end = x_slices[i], x_slices[i + 1]
            # Edge slice
            e_start, e_end = edge_slices[i], edge_slices[i + 1]

            x = x_all[n_start:n_end]
            pos = pos_all[n_start:n_end]

            edge_index = edge_index_all[:, e_start:e_end]
            edge_attr = edge_attr_all[e_start:e_end]
            edge_vector = edge_vector_all[e_start:e_end]

            y = y_all[i].view(1, -1)
            cell = cell_all[i].view(1, 3, 3)
            id_val = id_all[i]

            d = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_vector=edge_vector,
                y=y,
                pos=pos,
                cell=cell,
                id=int(id_val),
            )
            data_list.append(d)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    print("Initializing Datasets...")

    train_dataset = CrystalDataset(
        Config.TRAIN_METADATA_PATH,
        split_name="train",
        load_cached_data=load_cached_data,
    )

    val_dataset = CrystalDataset(
        Config.VAL_METADATA_PATH, split_name="val", load_cached_data=load_cached_data
    )

    test_dataset = CrystalDataset(
        Config.TEST_METADATA_PATH, split_name="test", load_cached_data=load_cached_data
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create DataLoaders
    # PyG DataLoader handles batching of graphs automatically
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
