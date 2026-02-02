import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config
from library.utils import GaussianSmearing, Standardizer


class CrystalGraphDataset(Dataset):
    """
    Dataset class for Crystal Graph Convolutional Neural Networks.
    Handles loading of metadata, parsing of geometry files, graph construction,
    and feature extraction (including RBF expansion and global feature standardization).
    """

    def __init__(
        self,
        metadata_path,
        split="train",
        load_cached_data=True,
        scaler_path=None,
        debug=False,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): Dataset split ('train', 'val', or 'test').
            load_cached_data (bool): Whether to try loading processed data from cache.
            scaler_path (str): Path to save/load the global feature standardizer.
            debug (bool): If True, load only a small subset of data for debugging.
        """
        self.metadata_path = metadata_path
        self.split = split
        self.load_cached_data = load_cached_data
        self.scaler_path = (
            scaler_path
            if scaler_path
            else os.path.join(Config.CACHE_DIR, "scalers.npz")
        )
        self.debug = debug

        # Load metadata
        self.df = pd.read_csv(metadata_path)
        if self.debug:
            self.df = self.df.iloc[:50]

        # Initialize RBF expander for edge features
        self.rbf_expander = GaussianSmearing(
            start=0.0,
            stop=Config.RADIUS,
            n_gaussians=Config.RBF_N_BINS,
            trainable=False,
        )

        # Atom type mapping for node features
        self.atom_type_map = {symbol: i for i, symbol in enumerate(Config.ATOM_TYPES)}

        # Process or load data
        self.data_list = self._get_data()

        # Handle Standardization of Global Features
        self.global_scaler = Standardizer(device="cpu")

        # Collect all global features to fit/transform
        if len(self.data_list) > 0:
            all_globals = torch.stack([d.global_feat for d in self.data_list])

            if self.split == "train":
                print("Fitting standardizer on training global features...")
                self.global_scaler.fit(all_globals)
                # Save scaler state
                if not self.debug:
                    self.global_scaler.save(self.scaler_path)
            else:
                # Load scaler state
                if os.path.exists(self.scaler_path):
                    print(f"Loading standardizer from {self.scaler_path}...")
                    self.global_scaler.load(self.scaler_path)
                else:
                    print(
                        f"Warning: Scaler file not found at {self.scaler_path}. Fitting on current data (suboptimal for val/test)."
                    )
                    self.global_scaler.fit(all_globals)

            # Apply standardization to all data in memory
            normalized_globals = self.global_scaler.transform(all_globals)
            for i, data in enumerate(self.data_list):
                # Replace global features with standardized version
                data.global_feat = normalized_globals[i].unsqueeze(0)  # Shape (1, 10)

    def _get_data(self):
        """
        Loads data from cache or processes it from scratch.
        """
        cache_file = os.path.join(
            Config.CACHE_DIR, f"{self.split}_graphs{'_debug' if self.debug else ''}.npz"
        )

        if self.load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached {self.split} data from {cache_file}...")
            return self._load_from_cache(cache_file)

        print(f"Processing {self.split} data from raw files...")
        data_list = []

        # Columns for global features (10 features)
        global_cols = [
            "lattice_vector_1_ang",
            "lattice_vector_2_ang",
            "lattice_vector_3_ang",
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
            "number_of_total_atoms",
            "percent_atom_al",
            "percent_atom_ga",
            "percent_atom_in",
        ]

        for _, row in self.df.iterrows():
            # Geometry file path
            xyz_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Read atoms using ASE
            try:
                atoms = read(xyz_path)
            except Exception as e:
                print(f"Error reading {xyz_path}: {e}")
                continue

            # Node Features: One-hot encoding of atom types
            symbols = atoms.get_chemical_symbols()
            x = torch.zeros((len(symbols), len(Config.ATOM_TYPES)), dtype=torch.float)
            for i, s in enumerate(symbols):
                if s in self.atom_type_map:
                    x[i, self.atom_type_map[s]] = 1.0

            # Edge Features: Neighbor list with Periodic Boundary Conditions
            # i: center indices, j: neighbor indices, d: distances
            i_indices, j_indices, d_values = neighbor_list("ijd", atoms, Config.RADIUS)

            # Convert to tensors
            edge_index = torch.tensor(
                np.vstack((i_indices, j_indices)), dtype=torch.long
            )
            edge_dist = torch.tensor(d_values, dtype=torch.float)

            # Compute RBF expansion for edge attributes
            edge_attr = self.rbf_expander(edge_dist)

            # Global Features
            glob_feat = torch.tensor(
                row[global_cols].values.astype(np.float32), dtype=torch.float
            )

            # Targets
            if self.split != "test":
                y = torch.tensor(
                    row[Config.TARGET_COLS].values.astype(np.float32), dtype=torch.float
                ).view(1, -1)
            else:
                # Dummy target for test set
                y = torch.zeros((1, len(Config.TARGET_COLS)), dtype=torch.float)

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feat=glob_feat,
                y=y,
                id=torch.tensor([row["id"]], dtype=torch.long),
            )

            data_list.append(data)

        # Save to cache
        # We save even in debug mode to a separate debug file
        self._save_to_cache(data_list, cache_file)

        return data_list

    def _save_to_cache(self, data_list, path):
        """
        Saves a list of Data objects to a .npz file without using pickle.
        """
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        global_feat_list = []
        y_list = []
        id_list = []

        num_nodes_list = []
        num_edges_list = []

        for data in data_list:
            x_list.append(data.x.numpy())
            edge_index_list.append(data.edge_index.numpy())
            edge_attr_list.append(data.edge_attr.numpy())
            global_feat_list.append(data.global_feat.numpy())
            y_list.append(data.y.numpy())
            id_list.append(data.id.numpy())

            num_nodes_list.append(data.x.shape[0])
            num_edges_list.append(data.edge_index.shape[1])

        # Concatenate arrays
        if not x_list:
            return

        x_all = np.concatenate(x_list, axis=0)
        edge_index_all = np.concatenate(edge_index_list, axis=1)  # (2, Total_Edges)
        edge_attr_all = np.concatenate(edge_attr_list, axis=0)
        global_feat_all = np.stack(global_feat_list, axis=0)  # (N_graphs, Feat)
        y_all = np.concatenate(y_list, axis=0)
        id_all = np.concatenate(id_list, axis=0)

        # Create slice indices for reconstruction
        slices_nodes = np.cumsum([0] + num_nodes_list)
        slices_edges = np.cumsum([0] + num_edges_list)

        np.savez(
            path,
            x=x_all,
            edge_index=edge_index_all,
            edge_attr=edge_attr_all,
            global_feat=global_feat_all,
            y=y_all,
            id=id_all,
            slices_nodes=slices_nodes,
            slices_edges=slices_edges,
        )
        print(f"Saved cache to {path}")

    def _load_from_cache(self, path):
        """
        Loads Data objects from a .npz file.
        """
        data = np.load(path)
        x_all = torch.from_numpy(data["x"])
        edge_index_all = torch.from_numpy(data["edge_index"])
        edge_attr_all = torch.from_numpy(data["edge_attr"])
        global_feat_all = torch.from_numpy(data["global_feat"])
        y_all = torch.from_numpy(data["y"])
        id_all = torch.from_numpy(data["id"])
        slices_nodes = data["slices_nodes"]
        slices_edges = data["slices_edges"]

        data_list = []
        num_graphs = len(slices_nodes) - 1

        for i in range(num_graphs):
            # Reconstruct Nodes
            start_n, end_n = slices_nodes[i], slices_nodes[i + 1]
            x = x_all[start_n:end_n]

            # Reconstruct Edges
            start_e, end_e = slices_edges[i], slices_edges[i + 1]
            edge_index = edge_index_all[:, start_e:end_e]
            edge_attr = edge_attr_all[start_e:end_e]

            # Reconstruct Global & Target
            global_feat = global_feat_all[i]
            y = y_all[i].unsqueeze(0)
            id_val = id_all[i]

            # Ensure id is 1D tensor
            if id_val.ndim == 0:
                id_val = id_val.unsqueeze(0)

            d = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feat=global_feat,
                y=y,
                id=id_val,
            )
            data_list.append(d)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_batch(batch):
    """
    Collate function for PyTorch Geometric Data objects.
    Combines a list of Data objects into a single Batch object.
    """
    return Batch.from_data_list(batch)
