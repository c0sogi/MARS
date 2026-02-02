import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.graph_utils import (
    load_structure,
    get_pbc_neighbor_graph,
    get_global_features,
)


class CrystalDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_path,
        global_scaler=None,
        target_scaler=None,
        load_cached_data=True,
        is_test=False,
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.cache_path = cache_path
        self.global_scaler = global_scaler
        self.target_scaler = target_scaler
        self.is_test = is_test
        self.data_list = []

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Caching Logic
        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                self._load_from_cache()
                loaded = True
                print(f"Loaded data from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        if not loaded:
            self._process_data()
            self._save_to_cache()
            print(f"Processed and saved data to cache: {cache_path}")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]

        # Apply scaling on the fly
        # Clone to avoid modifying the cached object in memory
        data = data.clone()

        if self.global_scaler is not None:
            # global_features is (1, 10)
            feat_np = data.global_features.numpy()
            if feat_np.ndim == 1:
                feat_np = feat_np.reshape(1, -1)
            scaled_feat = self.global_scaler.transform(feat_np)
            # Maintain 2D shape (1, 10)
            data.global_features = torch.tensor(scaled_feat, dtype=torch.float32)

        if self.target_scaler is not None and not self.is_test:
            # y is (1, 2)
            y_np = data.y.numpy()
            if y_np.ndim == 1:
                y_np = y_np.reshape(1, -1)
            scaled_y = self.target_scaler.transform(y_np)
            # Maintain 2D shape (1, 2)
            data.y = torch.tensor(scaled_y, dtype=torch.float32)

        return data

    def _process_data(self):
        df = pd.read_csv(self.metadata_path)
        data_list = []

        for _, row in df.iterrows():
            # Load structure
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            atoms = load_structure(file_path)

            if atoms is None:
                continue

            # Build Graph
            edge_index, edge_distances, atom_numbers = get_pbc_neighbor_graph(
                atoms, cutoff=Config.CUTOFF_RADIUS
            )

            # Global Features (10,) -> (1, 10)
            global_features = get_global_features(atoms).unsqueeze(0)

            # Targets (2,) -> (1, 2)
            if not self.is_test:
                y = torch.tensor(
                    [[row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]],
                    dtype=torch.float32,
                )
            else:
                # Dummy target for test set
                y = torch.zeros((1, 2), dtype=torch.float32)

            # Create PyG Data object
            # x will be atom numbers (embedding lookup index)
            data = Data(
                x=atom_numbers,
                edge_index=edge_index,
                edge_attr=edge_distances,
                global_features=global_features,
                y=y,
                id=torch.tensor([row["id"]], dtype=torch.long),
            )
            data_list.append(data)

        self.data_list = data_list

    def _save_to_cache(self):
        # Flatten data for np.savez to avoid pickle
        if not self.data_list:
            return

        # Collect all arrays
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        all_global = []
        all_y = []
        all_ids = []

        node_ptr = [0]
        edge_ptr = [0]

        for data in self.data_list:
            all_x.append(data.x.numpy())
            all_edge_index.append(data.edge_index.numpy())
            all_edge_attr.append(data.edge_attr.numpy())
            all_global.append(data.global_features.numpy())
            all_y.append(data.y.numpy())
            all_ids.append(data.id.numpy())

            node_ptr.append(node_ptr[-1] + data.x.shape[0])
            edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

        # Concatenate
        np_x = np.concatenate(all_x)
        np_edge_index = np.concatenate(all_edge_index, axis=1)  # (2, Total_Edges)
        np_edge_attr = np.concatenate(all_edge_attr)
        # Use concatenate for 2D inputs to maintain (N, F) structure
        np_global = np.concatenate(all_global, axis=0)
        np_y = np.concatenate(all_y, axis=0)
        np_ids = np.concatenate(all_ids)
        np_node_ptr = np.array(node_ptr)
        np_edge_ptr = np.array(edge_ptr)

        np.savez(
            self.cache_path,
            x=np_x,
            edge_index=np_edge_index,
            edge_attr=np_edge_attr,
            global_features=np_global,
            y=np_y,
            ids=np_ids,
            node_ptr=np_node_ptr,
            edge_ptr=np_edge_ptr,
        )

    def _load_from_cache(self):
        data = np.load(self.cache_path)

        x = torch.from_numpy(data["x"])
        edge_index = torch.from_numpy(data["edge_index"])
        edge_attr = torch.from_numpy(data["edge_attr"])
        global_features = torch.from_numpy(data["global_features"])
        y = torch.from_numpy(data["y"])
        ids = torch.from_numpy(data["ids"])
        node_ptr = data["node_ptr"]
        edge_ptr = data["edge_ptr"]

        data_list = []
        num_graphs = len(node_ptr) - 1

        for i in range(num_graphs):
            # Slice nodes
            n_start, n_end = node_ptr[i], node_ptr[i + 1]
            x_i = x[n_start:n_end]

            # Slice edges
            e_start, e_end = edge_ptr[i], edge_ptr[i + 1]
            edge_index_i = edge_index[:, e_start:e_end]
            edge_attr_i = edge_attr[e_start:e_end]

            # Slice others
            # Unsqueeze to restore (1, F) from (F,) slice if needed
            global_i = global_features[i]
            if global_i.dim() == 1:
                global_i = global_i.unsqueeze(0)

            y_i = y[i]
            if y_i.dim() == 1:
                y_i = y_i.unsqueeze(0)

            id_i = ids[i].unsqueeze(0)  # Keep as tensor

            d = Data(
                x=x_i,
                edge_index=edge_index_i,
                edge_attr=edge_attr_i,
                global_features=global_i,
                y=y_i,
                id=id_i,
            )
            data_list.append(d)

        self.data_list = data_list


def get_scalers(train_metadata_path):
    """
    Computes StandardScalers for global features and targets based on training data.
    """
    df = pd.read_csv(train_metadata_path)

    # 1. Fit Global Features Scaler
    # We need to compute global features for all train items to fit the scaler correctly.
    # To avoid re-parsing everything just for scaling if cache exists, we could check cache.
    # However, for robustness, we'll quickly extract them or rely on the Dataset to handle it.
    # Strategy: Instantiate a temporary Dataset without scalers to get the raw data tensors.

    temp_cache_path = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    # We assume training data processing might be needed.
    # If cache exists, it loads fast. If not, it processes.
    temp_dataset = CrystalDataset(
        metadata_path=train_metadata_path,
        cache_path=temp_cache_path,
        load_cached_data=True,
        is_test=False,
    )

    # Collect all global features and targets
    all_globals = []
    all_targets = []

    for data in temp_dataset:
        all_globals.append(data.global_features.numpy())
        all_targets.append(data.y.numpy())

    all_globals = np.vstack(all_globals)
    all_targets = np.vstack(all_targets)

    global_scaler = StandardScaler()
    global_scaler.fit(all_globals)

    target_scaler = StandardScaler()
    target_scaler.fit(all_targets)

    return global_scaler, target_scaler


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    # 1. Get Scalers from Training Data
    print("Computing/Loading scalers from training data...")
    global_scaler, target_scaler = get_scalers(Config.TRAIN_METADATA_PATH)

    # 2. Create Datasets
    print("Initializing Datasets...")

    train_dataset = CrystalDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=os.path.join(Config.CACHE_DIR, "train_graphs.npz"),
        global_scaler=global_scaler,
        target_scaler=target_scaler,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=os.path.join(Config.CACHE_DIR, "val_graphs.npz"),
        global_scaler=global_scaler,
        target_scaler=target_scaler,  # Val targets also scaled for loss computation
        load_cached_data=load_cached_data,
        is_test=False,
    )

    test_dataset = CrystalDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=os.path.join(Config.CACHE_DIR, "test_graphs.npz"),
        global_scaler=global_scaler,
        target_scaler=None,  # No targets for test
        load_cached_data=load_cached_data,
        is_test=True,
    )

    # 3. Create Loaders
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

    return train_loader, val_loader, test_loader, target_scaler
