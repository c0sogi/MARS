import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import ase.io
from ase.neighborlist import neighbor_list
from library.config import Config


class CrystalGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for Crystal Graphs.
    Reads geometry from XYZ files and constructs graphs with PBC awareness.
    Implements caching using .npz files to avoid pickle.
    """

    def __init__(
        self,
        metadata_path,
        root_dir=Config.INPUT_DIR,
        cache_path=None,
        load_cached=True,
        mode="train",
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the raw input files.
            cache_path (str): Path to save/load the processed .npz cache.
            load_cached (bool): Whether to attempt loading from cache.
            mode (str): 'train', 'val', or 'test'. Determines if targets are loaded.
        """
        self.metadata_path = metadata_path
        self.root_dir = root_dir
        self.cache_path = cache_path
        self.load_cached = load_cached
        self.mode = mode

        # Load metadata to get file paths and targets
        self.df = pd.read_csv(metadata_path)

        self.data_list = []
        self._process()

    def _process(self):
        """
        Internal method to load data either from cache or from raw files.
        """
        if self.load_cached and self.cache_path and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached data from {self.cache_path}...")
                self._load_from_cache()
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing from scratch.")

        print(f"Processing {len(self.df)} structures from raw files...")
        self.data_list = []

        for idx, row in self.df.iterrows():
            # Construct full file path
            # The metadata file_path is relative to input dir, e.g., "train/1/geometry.xyz"
            file_path = os.path.join(self.root_dir, row["file_path"])
            material_id = row["id"]

            # Load targets if available
            y = None
            if self.mode in ["train", "val"]:
                # Targets: formation_energy, bandgap_energy
                targets = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                y = torch.tensor([targets], dtype=torch.float32)

            # Process geometry
            data = self._process_one_structure(file_path, material_id, y)
            self.data_list.append(data)

        if self.cache_path:
            self._save_to_cache()

    def _process_one_structure(self, file_path, material_id, y):
        """
        Reads an XYZ file and converts it to a PyG Data object.
        """
        # Read structure using ASE
        atoms = ase.io.read(file_path)

        # Node Features: Atomic Numbers
        # We map atomic numbers to a continuous range if needed, or use them directly.
        # Here we use atomic numbers directly. Embedding layer will handle mapping.
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

        # Edge Features: Distance based on PBC
        # neighbor_list returns indices of interacting atoms (i, j) and distances (d)
        # 'd' returns distance values directly.
        # self_interaction=False ensures no loops.
        cutoff = Config.CUTOFF
        i_indices, j_indices, distances = neighbor_list(
            "ijd", atoms, cutoff, self_interaction=False
        )

        edge_index = torch.tensor(np.vstack((i_indices, j_indices)), dtype=torch.long)
        edge_attr = torch.tensor(distances, dtype=torch.float32).unsqueeze(
            1
        )  # (num_edges, 1)

        # Create Data object
        data = Data(
            x=atomic_numbers,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            material_id=torch.tensor([material_id], dtype=torch.long),
            num_nodes=len(atoms),
        )

        return data

    def _save_to_cache(self):
        """
        Saves the list of Data objects to a .npz file using numpy arrays.
        Avoids pickle.
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Aggregate all data into arrays
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        y_list = []
        id_list = []

        # Pointers to reconstruct individual graphs
        # node_ptr: start index of nodes for each graph
        # edge_ptr: start index of edges for each graph
        node_ptr = [0]
        edge_ptr = [0]

        for data in self.data_list:
            x_list.append(data.x.numpy())
            edge_index_list.append(data.edge_index.numpy())
            edge_attr_list.append(data.edge_attr.numpy())
            id_list.append(data.material_id.numpy())

            node_ptr.append(node_ptr[-1] + data.num_nodes)
            edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

            if data.y is not None:
                y_list.append(data.y.numpy())

        # Concatenate
        x_all = np.concatenate(x_list)
        edge_index_all = np.concatenate(edge_index_list, axis=1)
        edge_attr_all = np.concatenate(edge_attr_list)
        ids_all = np.concatenate(id_list)
        node_ptr = np.array(node_ptr, dtype=np.int64)
        edge_ptr = np.array(edge_ptr, dtype=np.int64)

        save_dict = {
            "x": x_all,
            "edge_index": edge_index_all,
            "edge_attr": edge_attr_all,
            "ids": ids_all,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
        }

        if y_list:
            y_all = np.concatenate(y_list)
            save_dict["y"] = y_all

        np.savez_compressed(self.cache_path, **save_dict)
        print(f"Data cached to {self.cache_path}")

    def _load_from_cache(self):
        """
        Loads data from .npz file and reconstructs Data objects.
        """
        data_dict = np.load(self.cache_path)

        x_all = torch.from_numpy(data_dict["x"])
        edge_index_all = torch.from_numpy(data_dict["edge_index"])
        edge_attr_all = torch.from_numpy(data_dict["edge_attr"])
        ids_all = torch.from_numpy(data_dict["ids"])
        node_ptr = data_dict["node_ptr"]
        edge_ptr = data_dict["edge_ptr"]

        has_y = "y" in data_dict
        if has_y:
            y_all = torch.from_numpy(data_dict["y"])

        self.data_list = []
        num_graphs = len(node_ptr) - 1

        for i in range(num_graphs):
            # Slice node features
            n_start, n_end = node_ptr[i], node_ptr[i + 1]
            x = x_all[n_start:n_end]

            # Slice edge features
            e_start, e_end = edge_ptr[i], edge_ptr[i + 1]
            edge_index = edge_index_all[:, e_start:e_end]
            edge_attr = edge_attr_all[e_start:e_end]

            # Material ID
            material_id = ids_all[i].unsqueeze(0)

            # Target
            y = y_all[i].unsqueeze(0) if has_y else None

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                material_id=material_id,
                num_nodes=n_end - n_start,
            )
            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached=True
):
    """
    Creates train, validation, and test dataloaders.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        load_cached (bool): Whether to load from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Train Dataset
    train_cache = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    train_dataset = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA,
        cache_path=train_cache,
        load_cached=load_cached,
        mode="train",
    )

    # Validation Dataset
    val_cache = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    val_dataset = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA,
        cache_path=val_cache,
        load_cached=load_cached,
        mode="val",
    )

    # Test Dataset
    test_cache = os.path.join(Config.CACHE_DIR, "test_graphs.npz")
    test_dataset = CrystalGraphDataset(
        metadata_path=Config.TEST_METADATA,
        cache_path=test_cache,
        load_cached=load_cached,
        mode="test",
    )

    # Create Loaders
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

    return train_loader, val_loader, test_loader
