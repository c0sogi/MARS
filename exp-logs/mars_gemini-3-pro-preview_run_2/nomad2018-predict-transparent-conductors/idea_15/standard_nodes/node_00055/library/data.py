import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import Standardizer


def get_pbc_graph(file_path, material_id, targets=None):
    """
    Converts a structure file into a PyTorch Geometric Data object.

    Args:
        file_path (str): Path to the geometry.xyz file.
        material_id (int/str): ID of the material.
        targets (list/np.array, optional): Target values [formation_energy, bandgap].

    Returns:
        torch_geometric.data.Data: Graph object with attributes:
            - x: Atomic numbers (node features)
            - edge_index: Graph connectivity
            - edge_attr: Interatomic distances
            - global_x: Global features (lattice params + composition)
            - y: Targets
    """
    # Load structure
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    # Explicitly specify format='aims' because the file extension .xyz is misleading. Cite debug_lesson_1
    atoms = read(full_path, format="aims")

    # 1. Node Features: Atomic Numbers
    # We map atomic numbers to a 0-based index or keep them as is.
    # Config.ATOM_INPUT_DIM suggests we can use atomic numbers directly if embedding layer handles it,
    # or we subtract 1 if the embedding expects 0-indexed.
    # Usually embedding layers handle up to max_index. Atomic numbers are 1-based.
    # We will use atomic numbers directly (Z).
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # 2. Edges: PBC-aware neighbor search
    # 'i': source index, 'j': target index, 'd': distance
    cut = Config.CUTOFF_RADIUS
    i, j, d = neighbor_list("ijd", atoms, cut)

    edge_index = torch.stack([torch.from_numpy(i), torch.from_numpy(j)], dim=0).long()
    edge_attr = torch.from_numpy(d).float().unsqueeze(1)  # Shape (num_edges, 1)

    # 3. Global Features
    # Lattice parameters: a, b, c, alpha, beta, gamma
    cell_params = atoms.get_cell_lengths_and_angles()

    # Composition fractions
    chemical_symbols = atoms.get_chemical_symbols()
    n_atoms = len(chemical_symbols)
    # Count specific elements relevant to the dataset description (Al, Ga, In)
    # Note: O is implicit remainder or explicit. We'll track Al, Ga, In as requested.
    n_al = chemical_symbols.count("Al")
    n_ga = chemical_symbols.count("Ga")
    n_in = chemical_symbols.count("In")

    comp_fracs = [n_al / n_atoms, n_ga / n_atoms, n_in / n_atoms]

    # Combine into a single vector of size 9
    global_feats = np.concatenate([cell_params, comp_fracs])
    global_x = torch.from_numpy(global_feats).float().unsqueeze(0)  # Shape (1, 9)

    # 4. Targets
    y = None
    if targets is not None:
        y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)  # Shape (1, 2)

    data = Data(
        x=atomic_numbers,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_x=global_x,
        y=y,
        material_id=material_id,
    )

    return data


class CrystalDataset(Dataset):
    def __init__(
        self, metadata_df, split_name, load_cached_data=True, root=Config.WORKING_DIR
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata.
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache.
            root (str): Root directory for cache.
        """
        self.metadata_df = metadata_df
        self.split_name = split_name
        self.load_cached_data = load_cached_data
        self.root = root
        self.cache_path = os.path.join(self.root, f"{split_name}_graphs.npz")

        # In-memory data list
        self._data_list = []

        super().__init__(root, transform=None, pre_transform=None)

        # Trigger processing or loading
        self._process_or_load()

    @property
    def processed_file_names(self):
        return [f"{self.split_name}_graphs.npz"]

    def _process_or_load(self):
        if self.load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading {self.split_name} data from cache: {self.cache_path}")
            try:
                self._load_from_cache()
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print(f"Processing {self.split_name} data...")
        self._process_data()
        print(f"Saving {self.split_name} data to cache: {self.cache_path}")
        self._save_to_cache()

    def _process_data(self):
        data_list = []

        # Debug limit
        if Config.DEBUG_DATA_LIMIT:
            df = self.metadata_df.head(Config.DEBUG_DATA_LIMIT)
        else:
            df = self.metadata_df

        for idx, row in df.iterrows():
            # Extract targets if available
            targets = None
            if all(c in row for c in Config.TARGET_COLS):
                targets = row[Config.TARGET_COLS].values.astype(float)

            # Construct graph
            data = get_pbc_graph(row["file_path"], row["id"], targets)
            data_list.append(data)

        self._data_list = data_list

    def _save_to_cache(self):
        # Flatten data for npz storage to avoid pickle
        # We need to store variable length arrays.
        # Strategy: Concatenate all arrays and store pointers.

        all_x = []
        all_edge_index = []
        all_edge_attr = []
        all_global_x = []
        all_y = []
        all_ids = []

        node_ptr = [0]
        edge_ptr = [0]

        for data in self._data_list:
            all_x.append(data.x.numpy())
            all_edge_index.append(data.edge_index.numpy())
            all_edge_attr.append(data.edge_attr.numpy())
            all_global_x.append(data.global_x.numpy())
            if data.y is not None:
                all_y.append(data.y.numpy())
            else:
                # Placeholder for test set if y is missing
                all_y.append(np.full((1, 2), np.nan))
            all_ids.append(data.material_id)

            node_ptr.append(node_ptr[-1] + data.x.shape[0])
            edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

        np.savez(
            self.cache_path,
            x=np.concatenate(all_x),
            edge_index=np.concatenate(all_edge_index, axis=1),
            edge_attr=np.concatenate(all_edge_attr, axis=0),
            global_x=np.concatenate(all_global_x, axis=0),
            y=np.concatenate(all_y, axis=0),
            ids=np.array(all_ids),
            node_ptr=np.array(node_ptr),
            edge_ptr=np.array(edge_ptr),
        )

    def _load_from_cache(self):
        data = np.load(self.cache_path)

        x_all = data["x"]
        edge_index_all = data["edge_index"]
        edge_attr_all = data["edge_attr"]
        global_x_all = data["global_x"]
        y_all = data["y"]
        ids_all = data["ids"]
        node_ptr = data["node_ptr"]
        edge_ptr = data["edge_ptr"]

        data_list = []
        num_graphs = len(ids_all)

        for i in range(num_graphs):
            # Slice nodes
            n_start, n_end = node_ptr[i], node_ptr[i + 1]
            x = torch.from_numpy(x_all[n_start:n_end]).long()

            # Slice edges
            e_start, e_end = edge_ptr[i], edge_ptr[i + 1]
            edge_index = torch.from_numpy(edge_index_all[:, e_start:e_end]).long()
            edge_attr = torch.from_numpy(edge_attr_all[e_start:e_end]).float()

            # Slice globals and targets
            global_x = torch.from_numpy(global_x_all[i]).float().unsqueeze(0)
            y = torch.from_numpy(y_all[i]).float().unsqueeze(0)

            # Handle NaN y for test set
            if torch.isnan(y).any():
                y = None

            d = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_x=global_x,
                y=y,
                material_id=ids_all[i],
            )
            data_list.append(d)

        self._data_list = data_list

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Also handles standardization of global features and targets.

    Returns:
        train_loader, val_loader, test_loader, target_scaler
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    # 2. Create Datasets
    # This will trigger processing/loading from cache
    train_dataset = CrystalDataset(train_meta, "train", load_cached_data)
    val_dataset = CrystalDataset(val_meta, "val", load_cached_data)
    test_dataset = CrystalDataset(test_meta, "test", load_cached_data)

    # 3. Standardization
    # We need to standardize global_x and y based on training statistics.

    # Collect training data for fitting
    all_train_global_x = []
    all_train_y = []

    for data in train_dataset:
        all_train_global_x.append(data.global_x)
        all_train_y.append(data.y)

    all_train_global_x = torch.cat(all_train_global_x, dim=0)
    all_train_y = torch.cat(all_train_y, dim=0)

    # Fit Scalers
    global_scaler = Standardizer()
    global_scaler.fit(all_train_global_x)

    target_scaler = Standardizer()
    target_scaler.fit(all_train_y)

    # Save scalers for inference/resuming
    global_scaler.save(os.path.join(Config.WORKING_DIR, "global_scaler.npz"))
    target_scaler.save(os.path.join(Config.WORKING_DIR, "target_scaler.npz"))

    # Apply transformation to all datasets in-place
    def transform_dataset(dataset):
        for data in dataset:
            data.global_x = global_scaler.transform(data.global_x)
            if data.y is not None:
                data.y = target_scaler.transform(data.y)

    transform_dataset(train_dataset)
    transform_dataset(val_dataset)
    transform_dataset(test_dataset)

    # 4. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, target_scaler
