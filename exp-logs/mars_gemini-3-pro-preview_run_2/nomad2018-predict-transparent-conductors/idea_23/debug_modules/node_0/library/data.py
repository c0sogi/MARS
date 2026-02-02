import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import TargetScaler


def process_structure(file_path, cutoff, max_neighbors):
    """
    Reads an XYZ file and constructs graph features (x, edge_index, edge_attr).
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    try:
        # Read the structure
        atoms = read(full_path)
    except Exception as e:
        print(f"Failed to read {full_path}: {e}")
        return None, None, None

    # Node features: Atomic numbers
    z = atoms.get_atomic_numbers()
    x = torch.tensor(z, dtype=torch.long)

    # Edge features: Neighbor search with PBC
    # i: source, j: target, D: distance vector
    i, j, D = neighbor_list("ijD", atoms, cutoff)

    # Compute distances
    dist = np.linalg.norm(D, axis=1)

    # Pruning to max_neighbors
    if max_neighbors is not None and len(i) > 0:
        # Fast numpy pruning
        # Stack into (N, 3) matrix: [i, j, dist]
        edges = np.column_stack((i, j, dist))

        # Sort by source (col 0) and distance (col 2)
        # lexsort keys are (secondary, primary)
        sort_idx = np.lexsort((edges[:, 2], edges[:, 0]))
        edges_sorted = edges[sort_idx]

        # Identify counts per source node
        sources, counts = np.unique(edges_sorted[:, 0].astype(int), return_counts=True)

        keep_indices = []
        start_ptr = 0
        for count in counts:
            # Keep at most max_neighbors per node
            n_keep = min(count, max_neighbors)
            keep_indices.extend(range(start_ptr, start_ptr + n_keep))
            start_ptr += count

        edges_pruned = edges_sorted[keep_indices]

        i = edges_pruned[:, 0].astype(int)
        j = edges_pruned[:, 1].astype(int)
        dist = edges_pruned[:, 2]

    edge_index = torch.tensor(np.vstack((i, j)), dtype=torch.long)
    edge_attr = torch.tensor(dist, dtype=torch.float).unsqueeze(1)  # (E, 1)

    return x, edge_index, edge_attr


def save_graphs_to_npz(data_list, path):
    """
    Flattens a list of Data objects and saves them to an NPZ file without pickle.
    """
    all_x = []
    all_edge_src = []
    all_edge_dst = []
    all_edge_dist = []
    all_y = []
    all_ids = []
    node_counts = []
    edge_counts = []

    for data in data_list:
        all_x.append(data.x.numpy())
        if data.edge_index.shape[1] > 0:
            all_edge_src.append(data.edge_index[0].numpy())
            all_edge_dst.append(data.edge_index[1].numpy())
            all_edge_dist.append(data.edge_attr.numpy().flatten())

        if data.y is not None:
            all_y.append(data.y.numpy())
        else:
            # Placeholder for test set
            all_y.append(np.array([[-1.0, -1.0]]))

        all_ids.append(data.id)
        node_counts.append(data.num_nodes)
        edge_counts.append(data.num_edges)

    # Handle case where some lists might be empty if dataset is empty (unlikely)
    node_z_arr = np.concatenate(all_x) if all_x else np.array([])
    edge_src_arr = np.concatenate(all_edge_src) if all_edge_src else np.array([])
    edge_dst_arr = np.concatenate(all_edge_dst) if all_edge_dst else np.array([])
    edge_dist_arr = np.concatenate(all_edge_dist) if all_edge_dist else np.array([])
    targets_arr = np.vstack(all_y) if all_y else np.array([])

    np.savez(
        path,
        node_z=node_z_arr,
        edge_src=edge_src_arr,
        edge_dst=edge_dst_arr,
        edge_dist=edge_dist_arr,
        targets=targets_arr,
        ids=np.array(all_ids),
        node_counts=np.array(node_counts),
        edge_counts=np.array(edge_counts),
    )


def load_graphs_from_npz(path):
    """
    Reconstructs a list of Data objects from a flattened NPZ file.
    """
    data = np.load(path)
    node_z = data["node_z"]
    edge_src = data["edge_src"]
    edge_dst = data["edge_dst"]
    edge_dist = data["edge_dist"]
    targets = data["targets"]
    ids = data["ids"]
    node_counts = data["node_counts"]
    edge_counts = data["edge_counts"]

    data_list = []
    node_ptr = 0
    edge_ptr = 0

    for i in range(len(ids)):
        n_nodes = node_counts[i]
        n_edges = edge_counts[i]

        x = torch.tensor(node_z[node_ptr : node_ptr + n_nodes], dtype=torch.long)

        if n_edges > 0:
            src = torch.tensor(
                edge_src[edge_ptr : edge_ptr + n_edges], dtype=torch.long
            )
            dst = torch.tensor(
                edge_dst[edge_ptr : edge_ptr + n_edges], dtype=torch.long
            )
            edge_index = torch.stack([src, dst], dim=0)
            dist = torch.tensor(
                edge_dist[edge_ptr : edge_ptr + n_edges], dtype=torch.float
            ).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            dist = torch.empty((0, 1), dtype=torch.float)

        y_val = targets[i]
        # Check if it's a dummy target (test set)
        if np.all(y_val == -1.0):
            y = None
        else:
            y = torch.tensor(y_val, dtype=torch.float).unsqueeze(
                0
            )  # (1, 2) if loaded from vstack, already (2,) so unsqueeze

        # Fix shape if vstack made it (2,) instead of (1,2)
        if y is not None and y.dim() == 1:
            y = y.unsqueeze(0)

        graph = Data(x=x, edge_index=edge_index, edge_attr=dist, y=y, id=int(ids[i]))
        data_list.append(graph)

        node_ptr += n_nodes
        edge_ptr += n_edges

    return data_list


class CrystalGraphDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_path,
        scaler=None,
        load_cached_data=True,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(None, transform, pre_transform)
        self.metadata_path = metadata_path
        self.cache_path = cache_path
        self.scaler = scaler
        self.data_list = []

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached graphs from {cache_path}...")
                self.data_list = load_graphs_from_npz(cache_path)
                loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        if not loaded:
            print(f"Processing raw data from {metadata_path}...")
            self.process_raw_data()
            print(f"Saving processed graphs to {cache_path}...")
            save_graphs_to_npz(self.data_list, cache_path)

        # Apply scaling if provided
        if self.scaler is not None:
            self.apply_scaling()

    def process_raw_data(self):
        df = pd.read_csv(self.metadata_path)

        # Debug sampling
        if Config.DEBUG_SAMPLE_SIZE is not None:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        graphs = []
        for idx, row in df.iterrows():
            file_path = row["file_path"]
            mat_id = row["id"]

            # Targets
            if all(col in row for col in Config.TARGET_COLS):
                y = torch.tensor(
                    [row[col] for col in Config.TARGET_COLS], dtype=torch.float
                ).unsqueeze(0)
            else:
                y = None

            x, edge_index, edge_attr = process_structure(
                file_path, Config.CUTOFF, Config.MAX_NEIGHBORS
            )

            if x is not None:
                data = Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=mat_id
                )
                graphs.append(data)

        self.data_list = graphs

    def apply_scaling(self):
        # Normalize targets in memory
        for data in self.data_list:
            if data.y is not None:
                # data.y is (1, 2)
                y_np = data.y.numpy()
                y_scaled = self.scaler.transform(y_np)
                data.y = torch.tensor(y_scaled, dtype=torch.float)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_dataloaders(load_cached_data=True):
    """
    Creates dataloaders for train, val, and test sets.
    Fits the scaler on the training data.
    """
    # 1. Fit Scaler on Training Metadata
    print("Fitting target scaler on training metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if Config.DEBUG_SAMPLE_SIZE:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)

    y_train = train_df[Config.TARGET_COLS].values
    scaler = TargetScaler()
    scaler.fit(y_train)

    # Save scaler for inference later
    scaler.save(Config.TARGET_SCALER_CACHE)

    # 2. Create Datasets
    print("Creating Train Dataset...")
    train_dataset = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_GRAPHS_CACHE,
        scaler=scaler,
        load_cached_data=load_cached_data,
    )

    print("Creating Validation Dataset...")
    val_dataset = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.VAL_GRAPHS_CACHE,
        scaler=scaler,
        load_cached_data=load_cached_data,
    )

    print("Creating Test Dataset...")
    # Test dataset does not need scaler for y (as y is missing/dummy)
    test_dataset = CrystalGraphDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=Config.TEST_GRAPHS_CACHE,
        scaler=None,
        load_cached_data=load_cached_data,
    )

    # 3. Create Loaders
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

    return train_loader, val_loader, test_loader, scaler
