import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from ase.io import read
from library.config import Config
from library.utils import TargetScaler


class CrystalGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for crystal structures.
    Reads metadata, loads XYZ files, constructs graphs with distance-based edges,
    and handles caching of processed graphs.
    """

    def __init__(
        self,
        metadata_path,
        name,
        load_cached_data=True,
        scaler=None,
        fit_scaler=False,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            name (str): Name of the dataset split (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to try loading from cache.
            scaler (TargetScaler, optional): Scaler instance to normalize targets.
            fit_scaler (bool): Whether to fit the scaler on this dataset.
            transform (callable, optional): A function/transform that takes in an
                torch_geometric.data.Data object and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in
                an torch_geometric.data.Data object and returns a transformed version.
        """
        self.metadata_path = metadata_path
        self.name = name
        self.load_cached_data = load_cached_data
        self.scaler = scaler
        self.fit_scaler = fit_scaler

        # Define cache path in the working directory
        self.cache_path = os.path.join(Config.WORKING_DIR, f"{name}_graphs.npz")

        # Initialize InMemoryDataset (root is set to current dir as we handle paths manually)
        super().__init__(".", transform, pre_transform)

        # Load data into memory
        self.data, self.slices = self._load_data()

        # Handle Target Scaling
        if self.fit_scaler:
            self.scaler = TargetScaler()
            self.scaler.fit(self.data.y)

        if self.scaler is not None and self.data.y is not None:
            # Transform targets using the scaler (in-place modification of the big tensor)
            # Note: For test set, y is dummy, so scaling doesn't matter but we apply it for consistency
            self.data.y = self.scaler.transform(self.data.y)

    def _load_data(self):
        """
        Loads data from cache if available, otherwise processes raw files.
        """
        # 1. Try loading from cache
        if self.load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading {self.name} data from cache: {self.cache_path}")
            try:
                data_dict = np.load(self.cache_path)

                # Reconstruct tensors from numpy arrays
                x = torch.from_numpy(data_dict["x"])
                edge_index = torch.from_numpy(data_dict["edge_index"])
                edge_attr = torch.from_numpy(data_dict["edge_attr"])

                # Handle y (targets)
                if "y" in data_dict:
                    y = torch.from_numpy(data_dict["y"])
                else:
                    y = None

                # Reconstruct slices dictionary
                slices = {
                    "x": torch.from_numpy(data_dict["x_slice"]),
                    "edge_index": torch.from_numpy(data_dict["edge_index_slice"]),
                    "edge_attr": torch.from_numpy(data_dict["edge_attr_slice"]),
                    "y": torch.from_numpy(data_dict["y_slice"]),
                }

                # Create the monolithic Data object used by InMemoryDataset
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                return data, slices

            except Exception as e:
                print(
                    f"Failed to load cache for {self.name}: {e}. Reprocessing from raw files..."
                )

        # 2. Process from raw files
        print(f"Processing {self.name} data from raw files...")
        df = pd.read_csv(self.metadata_path)

        data_list = []

        for _, row in df.iterrows():
            # Construct full path to geometry file
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Read atomic structure using ASE
            # Explicitly specify format='aims' because the file extension .xyz is misleading
            atoms = read(full_path, format="aims")

            # Node features: Atomic numbers (LongTensor)
            atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

            # Positions (FloatTensor)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

            # Construct Graph Edges
            # radius_graph returns edge_index [2, num_edges]
            edge_index = radius_graph(
                pos,
                r=Config.CUTOFF_RADIUS,
                batch=None,
                loop=False,
                max_num_neighbors=Config.MAX_NEIGHBORS,
            )

            # Calculate Edge Features (Distances)
            row_idx, col_idx = edge_index
            dist = (pos[row_idx] - pos[col_idx]).norm(dim=-1).view(-1, 1)

            # Targets
            if all(col in row for col in Config.TARGET_COLS):
                # Training/Validation case: Load real targets
                y = torch.tensor(
                    [[row[col] for col in Config.TARGET_COLS]], dtype=torch.float
                )
            else:
                # Test case: Dummy targets
                y = torch.tensor([[0.0] * len(Config.TARGET_COLS)], dtype=torch.float)

            # Create Data object
            data = Data(x=atomic_numbers, edge_index=edge_index, edge_attr=dist, y=y)
            data_list.append(data)

        # Collate list of Data objects into a single Data object and slices
        data, slices = self.collate(data_list)

        # Save to cache
        # We convert tensors to numpy for storage in .npz
        save_dict = {
            "x": data.x.numpy(),
            "edge_index": data.edge_index.numpy(),
            "edge_attr": data.edge_attr.numpy(),
            "x_slice": slices["x"].numpy(),
            "edge_index_slice": slices["edge_index"].numpy(),
            "edge_attr_slice": slices["edge_attr"].numpy(),
            "y_slice": slices["y"].numpy(),
        }

        if data.y is not None:
            save_dict["y"] = data.y.numpy()

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(self.cache_path, **save_dict)
        print(f"Saved {self.name} data to cache: {self.cache_path}")

        return data, slices


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size for the loaders.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # 1. Train Dataset
    # We fit the scaler on the training data
    train_dataset = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        name="train",
        load_cached_data=load_cached_data,
        fit_scaler=True,
    )
    scaler = train_dataset.scaler

    # 2. Validation Dataset
    # We use the scaler fitted on training data
    val_dataset = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        name="val",
        load_cached_data=load_cached_data,
        scaler=scaler,
        fit_scaler=False,
    )

    # 3. Test Dataset
    # We use the scaler from training data (though targets are dummy, it keeps structure consistent)
    test_dataset = CrystalGraphDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        name="test",
        load_cached_data=load_cached_data,
        scaler=scaler,
        fit_scaler=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
