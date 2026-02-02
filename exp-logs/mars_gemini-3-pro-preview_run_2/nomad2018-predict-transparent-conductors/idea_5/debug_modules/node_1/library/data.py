import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import Config


class CrystalGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for Crystal Graphs with Global Conditioning.
    Handles loading from metadata/XYZ, graph construction with PBC, and caching.
    """

    def __init__(
        self,
        metadata_path,
        split="train",
        load_cached_data=True,
        transform=None,
        pre_transform=None,
    ):
        self.metadata_path = metadata_path
        self.split = split
        self.load_cached_data = load_cached_data

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Define cache file path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{split}_graphs.npz")

        super().__init__(Config.WORKING_DIR, transform, pre_transform)

        # Load data into memory
        self.data_list = self._load_or_process_data()

    def _parse_custom_xyz(self, path):
        """
        Parses the custom XYZ format provided in the dataset.
        Extracts lattice vectors from headers and atomic positions.
        """
        cell = []
        positions = []
        symbols = []

        # Full path handling
        full_path = os.path.join(Config.INPUT_DIR, path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Geometry file not found: {full_path}")

        with open(full_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "lattice_vector":
                    cell.append([float(x) for x in parts[1:4]])
                elif parts[0] == "atom":
                    positions.append([float(x) for x in parts[1:4]])
                    symbols.append(parts[4])

        # Create ASE Atoms object
        # We assume pbc=True for all crystals in this dataset
        if len(cell) == 3:
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        else:
            # Fallback if cell is missing (should not happen based on description)
            atoms = Atoms(symbols=symbols, positions=positions, pbc=True)

        return atoms

    def _get_pbc_neighbors(self, atoms):
        """
        Computes neighbors respecting periodic boundary conditions.
        Returns source indices, target indices, and distances.
        """
        # 'i' : source index
        # 'j' : target index
        # 'd' : distance
        i_indices, j_indices, distances = neighbor_list(
            "ijd", atoms, Config.CUTOFF_RADIUS
        )

        # Filter out self-loops with very small distance (same atom)
        # neighbor_list can return self-interactions across PBC boundaries (valid)
        # or the atom itself (invalid for graph convolution typically)
        # We filter out exact self-loops (distance ~ 0)
        mask = distances > 1e-4
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        distances = distances[mask]

        return i_indices, j_indices, distances

    def _process_single_entry(self, row):
        """
        Converts a single metadata row and its corresponding XYZ file into a PyG Data object.
        """
        # 1. Parse Structure
        file_path = row["file_path"]
        atoms = self._parse_custom_xyz(file_path)

        # 2. Graph Construction
        src, dst, dists = self._get_pbc_neighbors(atoms)

        # Edge Index [2, E]
        edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

        # Edge Attributes [E, 1] (Distances)
        edge_attr = torch.tensor(dists, dtype=torch.float).unsqueeze(1)

        # Node features: Atomic numbers [N]
        x = torch.tensor(atoms.numbers, dtype=torch.long)

        # 3. Global Features [1, G]
        # Extract columns defined in Config
        global_feats = row[Config.GLOBAL_FEATURE_COLS].values.astype(np.float32)
        global_feat_tensor = torch.tensor(global_feats, dtype=torch.float).unsqueeze(0)

        # 4. Targets [1, T] (if available)
        y = None
        if all(col in row for col in Config.TARGET_COLS):
            targets = row[Config.TARGET_COLS].values.astype(np.float32)
            y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)

        # 5. ID
        material_id = row["id"]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_feat=global_feat_tensor,
            y=y,
            id=material_id,
        )

        return data

    def _load_or_process_data(self):
        """
        Loads data from cache if available, otherwise processes from scratch and caches it.
        """
        # Try loading from cache
        if self.load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading {self.split} data from cache: {self.cache_path}")
            try:
                cached = np.load(self.cache_path)

                # Reconstruct Data objects from flattened arrays
                all_x = torch.from_numpy(cached["x"])
                all_edge_index = torch.from_numpy(cached["edge_index"])
                all_edge_attr = torch.from_numpy(cached["edge_attr"])
                all_global_feat = torch.from_numpy(cached["global_feat"])

                # Targets might be empty for test set
                if "y" in cached and cached["y"].size > 0:
                    all_y = torch.from_numpy(cached["y"])
                else:
                    all_y = None

                all_ids = cached["id"]

                node_splits = cached["node_splits"]
                edge_splits = cached["edge_splits"]

                data_list = []
                num_graphs = len(node_splits) - 1

                for i in range(num_graphs):
                    # Slice Nodes
                    n_start, n_end = node_splits[i], node_splits[i + 1]
                    x = all_x[n_start:n_end]

                    # Slice Edges
                    e_start, e_end = edge_splits[i], edge_splits[i + 1]
                    edge_index = all_edge_index[:, e_start:e_end]
                    edge_attr = all_edge_attr[e_start:e_end]

                    # Slice Global
                    global_feat = all_global_feat[i].unsqueeze(0)

                    # Slice Target
                    y = all_y[i].unsqueeze(0) if all_y is not None else None

                    # ID
                    mid = int(all_ids[i])

                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        global_feat=global_feat,
                        y=y,
                        id=mid,
                    )
                    data_list.append(data)

                return data_list

            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Processing {self.split} data...")
        df = pd.read_csv(self.metadata_path)

        # Optional: Subsample for debugging
        if Config.DEBUG:
            print(f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLE_SIZE} records.")
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        data_list = []
        for _, row in df.iterrows():
            try:
                data = self._process_single_entry(row)
                data_list.append(data)
            except Exception as e:
                print(f"Error processing id {row.get('id', 'unknown')}: {e}")

        # Save to cache
        self._save_to_cache(data_list)

        return data_list

    def _save_to_cache(self, data_list):
        """
        Saves the list of Data objects to a compressed numpy file (.npz)
        by flattening the tensors. Avoids pickle.
        """
        if not data_list:
            return

        # Flatten x (nodes)
        x_list = [d.x.numpy() for d in data_list]
        all_x = np.concatenate(x_list)
        node_splits = np.cumsum([0] + [len(x) for x in x_list])

        # Flatten edge_index
        edge_index_list = [d.edge_index.numpy() for d in data_list]
        if edge_index_list:
            all_edge_index = np.concatenate(edge_index_list, axis=1)
        else:
            all_edge_index = np.empty((2, 0), dtype=np.int64)
        edge_splits = np.cumsum([0] + [e.shape[1] for e in edge_index_list])

        # Flatten edge_attr
        edge_attr_list = [d.edge_attr.numpy() for d in data_list]
        if edge_attr_list:
            all_edge_attr = np.concatenate(edge_attr_list)
        else:
            all_edge_attr = np.empty((0, 1), dtype=np.float32)

        # Stack global_feat
        global_feat_list = [d.global_feat.numpy().flatten() for d in data_list]
        all_global_feat = np.stack(global_feat_list)

        # Stack y
        if data_list[0].y is not None:
            y_list = [d.y.numpy().flatten() for d in data_list]
            all_y = np.stack(y_list)
        else:
            all_y = np.array([])

        # Stack IDs
        all_ids = np.array([d.id for d in data_list])

        np.savez(
            self.cache_path,
            x=all_x,
            edge_index=all_edge_index,
            edge_attr=all_edge_attr,
            global_feat=all_global_feat,
            y=all_y,
            id=all_ids,
            node_splits=node_splits,
            edge_splits=edge_splits,
        )
        print(f"Saved processed data to {self.cache_path}")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_train_val_datasets(load_cached=True):
    """
    Returns train and validation datasets.
    """
    train_ds = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split="train",
        load_cached_data=load_cached,
    )

    val_ds = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split="val",
        load_cached_data=load_cached,
    )

    return train_ds, val_ds


def get_test_dataset(load_cached=True):
    """
    Returns test dataset.
    """
    test_ds = CrystalGraphDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        split="test",
        load_cached_data=load_cached,
    )
    return test_ds
