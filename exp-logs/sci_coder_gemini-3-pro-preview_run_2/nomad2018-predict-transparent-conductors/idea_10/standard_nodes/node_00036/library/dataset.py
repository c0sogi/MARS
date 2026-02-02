import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.data_utils import process_dataset, GaussianRBF, StandardScaler


class CrystalGraphDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_prefix="train",
        load_cached_data=True,
        global_scaler=None,
        target_scaler=None,
        fit_scalers=False,
        sample_limit=None,
    ):
        """
        PyTorch Dataset for Crystal Graphs.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            cache_prefix (str): Prefix for the cache file (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to load processed data from cache.
            global_scaler (StandardScaler, optional): Scaler for global features.
            target_scaler (StandardScaler, optional): Scaler for targets.
            fit_scalers (bool): Whether to fit the provided scalers on this dataset.
            sample_limit (int, optional): Limit the number of samples (for debugging).
        """
        self.metadata_df = pd.read_csv(metadata_path)

        # Process data (handles caching internally via data_utils)
        # This returns a dictionary with numpy arrays/lists
        self.data_dict = process_dataset(
            self.metadata_df,
            load_cached_data=load_cached_data,
            cache_prefix=cache_prefix,
        )

        # Unpack data
        self.atom_features_list = self.data_dict["atom_features_list"]
        self.edge_index_list = self.data_dict["edge_index_list"]
        self.edge_attr_list = self.data_dict["edge_attr_list"]  # Raw distances
        self.global_features = self.data_dict["global_features"]
        self.targets = self.data_dict["targets"]
        self.ids = self.data_dict["ids"]

        # Apply sample limit if requested (useful for quick debugging without reprocessing)
        if sample_limit is not None:
            limit = min(sample_limit, len(self.ids))
            self.atom_features_list = self.atom_features_list[:limit]
            self.edge_index_list = self.edge_index_list[:limit]
            self.edge_attr_list = self.edge_attr_list[:limit]
            self.global_features = self.global_features[:limit]
            self.ids = self.ids[:limit]
            if self.targets is not None:
                self.targets = self.targets[:limit]

        self.global_scaler = global_scaler
        self.target_scaler = target_scaler

        # Fit scalers if requested (typically only for the training set)
        if fit_scalers:
            if self.global_scaler is not None:
                self.global_scaler.fit(self.global_features)
            if self.target_scaler is not None and self.targets is not None:
                self.target_scaler.fit(self.targets)

        # Initialize Gaussian RBF expansion for edge distances
        # This is a static transformation, so we can init it here.
        self.rbf = GaussianRBF(
            start=0.0,
            stop=Config.CUTOFF_RADIUS,
            n_rbf=Config.N_RBF,
            sigma=Config.RBF_SIGMA,
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a torch_geometric Data object for the crystal.
        """
        # 1. Node Features (Atomic Numbers)
        # Shape: (num_nodes,)
        # We use LongTensor for embedding lookups
        x = torch.tensor(self.atom_features_list[idx], dtype=torch.long)

        # 2. Edge Index
        # Shape: (2, num_edges)
        edge_index = torch.tensor(self.edge_index_list[idx], dtype=torch.long)

        # 3. Edge Attributes (Distances -> RBF)
        # Distances shape: (num_edges,)
        dists = torch.tensor(self.edge_attr_list[idx], dtype=torch.float32)
        # Expand distances using Gaussian RBF
        # Shape: (num_edges, n_rbf)
        edge_attr = self.rbf(dists)

        # 4. Global Features
        # Shape: (global_dim,)
        g_feat = self.global_features[idx]
        if self.global_scaler is not None:
            g_feat = self.global_scaler.transform(g_feat)

        # Add batch dimension (1, global_dim) to align with PyG batching behavior
        # where global features are often concatenated.
        global_x = torch.tensor(g_feat, dtype=torch.float32).unsqueeze(0)

        # 5. Targets
        y = None
        if self.targets is not None:
            target_val = self.targets[idx]
            if self.target_scaler is not None:
                target_val = self.target_scaler.transform(target_val)
            # Shape: (1, num_targets)
            y = torch.tensor(target_val, dtype=torch.float32).unsqueeze(0)

        # 6. ID
        material_id = self.ids[idx]
        # Fix: Convert to tensor to ensure proper batching and .cpu() availability
        material_id = torch.tensor([material_id], dtype=torch.long)

        # Create PyG Data object
        # Note: PyG Batch.from_data_list will concatenate x, edge_index, edge_attr, global_x, and y
        # along the first dimension (except edge_index which gets offsets).
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_x=global_x,
            y=y,
            material_id=material_id,
        )

        return data


def collate_graphs(batch):
    """
    Collates a list of PyG Data objects into a single Batch object.
    This function is intended to be used as the `collate_fn` in a PyTorch DataLoader.
    """
    return Batch.from_data_list(batch)
