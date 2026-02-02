import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import ase.io
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import TargetScaler


class CrystalGraphDataset(Dataset):
    """
    Dataset class for crystal graphs.
    Loads atomic structures from XYZ files, constructs graphs with PBC,
    and caches them as compressed numpy files.
    """

    def __init__(self, metadata_path, cache_path, subset_name, load_cached_data=True):
        super().__init__()
        self.metadata_path = metadata_path
        self.cache_path = cache_path
        self.subset_name = subset_name
        self.load_cached_data = load_cached_data
        self.data_list = []

        # Ensure working directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self._load()

    def _load(self):
        loaded = False
        if self.load_cached_data and os.path.exists(self.cache_path):
            try:
                print(
                    f"Loading cached {self.subset_name} data from {self.cache_path}..."
                )
                data_dict = np.load(self.cache_path, allow_pickle=False)

                # Reconstruct Data objects from flattened arrays
                all_x = data_dict["x"]
                all_edge_index = data_dict["edge_index"]
                all_edge_attr = data_dict["edge_attr"]
                all_y = data_dict["y"]
                all_ids = data_dict["ids"]
                node_ptr = data_dict["node_ptr"]
                edge_ptr = data_dict["edge_ptr"]

                num_graphs = len(all_ids)
                self.data_list = []

                for i in range(num_graphs):
                    # Extract nodes for graph i
                    n_start, n_end = node_ptr[i], node_ptr[i + 1]
                    x = torch.tensor(all_x[n_start:n_end], dtype=torch.long)

                    # Extract edges for graph i
                    e_start, e_end = edge_ptr[i], edge_ptr[i + 1]
                    edge_index = torch.tensor(
                        all_edge_index[:, e_start:e_end], dtype=torch.long
                    )
                    edge_attr = torch.tensor(
                        all_edge_attr[e_start:e_end], dtype=torch.float
                    ).unsqueeze(1)

                    # Extract target and ID
                    y = torch.tensor(all_y[i], dtype=torch.float).unsqueeze(
                        0
                    )  # Shape (1, num_targets)
                    id_val = int(all_ids[i])

                    data = Data(
                        x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=id_val
                    )
                    self.data_list.append(data)

                print(f"Successfully loaded {len(self.data_list)} graphs.")
                loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                loaded = False

        if not loaded:
            print(f"Processing {self.subset_name} data from scratch...")
            self.process_data()
            self.save_cache()

    def process_data(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        data_list = []

        # Iterate over metadata rows
        for idx, row in df.iterrows():
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            mat_id = int(row["id"])

            # Determine targets
            if self.subset_name == "test":
                # Dummy targets for test set
                y = [0.0, 0.0]
            else:
                y = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]

            # Load atomic structure
            try:
                atoms = ase.io.read(full_path, format="aims")
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
                continue

            # Compute neighbors with PBC
            # cutoff is in Angstroms
            cut = Config.GRAPH_CUTOFF
            # neighbor_list returns i, j, d, D. We only need i, j, d.
            # i: source indices, j: target indices, d: distances
            i_indices, j_indices, distances = neighbor_list(
                "ijd", atoms, cut, self_interaction=False
            )

            # Create Node Features (Atomic Numbers)
            atomic_numbers = atoms.get_atomic_numbers()
            x = torch.tensor(atomic_numbers, dtype=torch.long)

            # Handle Max Neighbors Pruning
            # If a node has too many neighbors, keep the closest ones.
            if Config.MAX_NEIGHBORS > 0:
                num_nodes = len(atomic_numbers)
                new_i_list = []
                new_j_list = []
                new_dist_list = []

                # Group edges by source node to filter
                # This approach ensures we process per-node constraints
                for node_idx in range(num_nodes):
                    # Find edges where source is node_idx
                    mask = i_indices == node_idx

                    if not np.any(mask):
                        continue

                    node_js = j_indices[mask]
                    node_dists = distances[mask]

                    # Sort by distance
                    sorted_indices = np.argsort(node_dists)

                    # Keep top K
                    k = min(len(sorted_indices), Config.MAX_NEIGHBORS)
                    keep_indices = sorted_indices[:k]

                    new_i_list.extend([node_idx] * k)
                    new_j_list.extend(node_js[keep_indices])
                    new_dist_list.extend(node_dists[keep_indices])

                # Reconstruct edge tensors
                edge_index = torch.tensor([new_i_list, new_j_list], dtype=torch.long)
                edge_attr = torch.tensor(new_dist_list, dtype=torch.float).unsqueeze(1)
            else:
                # No pruning
                edge_index = torch.tensor(
                    np.vstack((i_indices, j_indices)), dtype=torch.long
                )
                edge_attr = torch.tensor(distances, dtype=torch.float).unsqueeze(1)

            y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(0)  # (1, 2)

            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor, id=mat_id
            )
            data_list.append(data)

        self.data_list = data_list

    def save_cache(self):
        print(f"Saving {self.subset_name} cache to {self.cache_path}...")

        if not self.data_list:
            print("Warning: No data to save!")
            return

        # Flatten data for efficient npz storage
        all_x = []
        all_edge_index_0 = []
        all_edge_index_1 = []
        all_edge_attr = []
        all_y = []
        all_ids = []

        node_ptr = [0]
        edge_ptr = [0]

        for data in self.data_list:
            all_x.append(data.x.numpy())
            all_edge_index_0.append(data.edge_index[0].numpy())
            all_edge_index_1.append(data.edge_index[1].numpy())
            all_edge_attr.append(data.edge_attr.numpy().flatten())
            all_y.append(data.y.numpy().flatten())
            all_ids.append(data.id)

            node_ptr.append(node_ptr[-1] + len(data.x))
            edge_ptr.append(edge_ptr[-1] + data.edge_index.shape[1])

        # Concatenate arrays
        x_concat = np.concatenate(all_x)
        if len(all_edge_index_0) > 0:
            edge_index_concat = np.vstack(
                (np.concatenate(all_edge_index_0), np.concatenate(all_edge_index_1))
            )
            edge_attr_concat = np.concatenate(all_edge_attr)
        else:
            # Handle case with no edges (unlikely but possible)
            edge_index_concat = np.empty((2, 0), dtype=np.int64)
            edge_attr_concat = np.empty((0,), dtype=np.float32)

        y_concat = np.array(all_y)
        ids_concat = np.array(all_ids)
        node_ptr_arr = np.array(node_ptr)
        edge_ptr_arr = np.array(edge_ptr)

        # Save compressed
        np.savez_compressed(
            self.cache_path,
            x=x_concat,
            edge_index=edge_index_concat,
            edge_attr=edge_attr_concat,
            y=y_concat,
            ids=ids_concat,
            node_ptr=node_ptr_arr,
            edge_ptr=edge_ptr_arr,
        )
        print("Cache saved.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_dataloaders(load_cached_data=True):
    """
    Creates dataloaders for train, val, and test sets.
    Fits and applies TargetScaler to train and val sets.
    """
    # 1. Initialize Datasets
    # This will load from cache or process from scratch
    train_dataset = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_GRAPHS_CACHE,
        subset_name="train",
        load_cached_data=load_cached_data,
    )

    val_dataset = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.VAL_GRAPHS_CACHE,
        subset_name="val",
        load_cached_data=load_cached_data,
    )

    test_dataset = CrystalGraphDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=Config.TEST_GRAPHS_CACHE,
        subset_name="test",
        load_cached_data=load_cached_data,
    )

    # 2. Fit and Apply Scaler
    # We need to collect all training targets to fit the scaler
    train_y_list = [data.y for data in train_dataset]

    if train_y_list:
        train_y_all = torch.cat(train_y_list, dim=0)  # Shape (N, 2)

        scaler = TargetScaler()
        scaler.fit(train_y_all)
        scaler.save(Config.TARGET_SCALER_CACHE)
        print("Target scaler fitted and saved.")
        print(f"  Mean: {scaler.mean}")
        print(f"  Std:  {scaler.std}")

        # Transform train and val targets in-memory
        # Note: We do not modify the cached files, only the objects in memory
        for data in train_dataset:
            data.y = scaler.transform(data.y)

        for data in val_dataset:
            data.y = scaler.transform(data.y)

        # Test targets are dummy (0.0), scaling them doesn't matter much but let's leave them
        # or scale them if consistency is needed. Since we don't evaluate on test y, it's fine.
    else:
        print("Warning: Training dataset is empty. Scaler not fitted.")

    # 3. Create DataLoaders
    # num_workers=0 is safer for simple scripts; increase if CPU bound
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
