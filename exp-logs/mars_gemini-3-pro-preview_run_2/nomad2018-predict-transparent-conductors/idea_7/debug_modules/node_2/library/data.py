import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SCALER_CACHE_PATH,
    BATCH_SIZE,
    SEED,
    NUM_GLOBAL_FEATURES,
    NUM_TARGETS,
)
from library.preprocessing import process_dataset


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for Crystal Structures.

    Loads preprocessed graph data and global features.
    Handles normalization of global features and targets.
    Returns torch_geometric.data.Data objects.
    """

    def __init__(
        self,
        metadata_path,
        cache_path,
        split="train",
        scalers=None,
        load_cached_data=True,
    ):
        super().__init__()
        self.split = split

        # Load data using the provided preprocessing function
        # This handles caching of the raw graph construction and feature extraction
        data_dict = process_dataset(
            metadata_path, cache_path, load_cached_data=load_cached_data
        )

        self.ids = data_dict["ids"]
        self.node_feats_list = data_dict["node_feats_list"]
        self.edge_index_list = data_dict["edge_index_list"]
        self.edge_dist_list = data_dict["edge_dist_list"]
        self.global_feats = data_dict["global_feats_list"]
        self.targets = data_dict["targets"]

        # Handle Normalization
        self.scalers = {}
        if split == "train":
            # Compute statistics for Global Features
            g_mean = np.mean(self.global_feats, axis=0)
            g_std = np.std(self.global_feats, axis=0)
            # Avoid division by zero
            g_std[g_std == 0] = 1.0

            self.scalers["global_mean"] = g_mean
            self.scalers["global_std"] = g_std

            # Compute statistics for Targets
            t_mean = np.mean(self.targets, axis=0)
            t_std = np.std(self.targets, axis=0)
            t_std[t_std == 0] = 1.0

            self.scalers["target_mean"] = t_mean
            self.scalers["target_std"] = t_std

            # Save scalers
            os.makedirs(os.path.dirname(SCALER_CACHE_PATH), exist_ok=True)
            np.save(SCALER_CACHE_PATH, self.scalers)

        else:
            # Load scalers
            if scalers is not None:
                self.scalers = scalers
            elif os.path.exists(SCALER_CACHE_PATH):
                self.scalers = np.load(SCALER_CACHE_PATH, allow_pickle=True).item()
            else:
                # Fallback if no scalers provided/found (e.g. inference without training first - though unlikely in this pipeline)
                # Initialize with identity (no scaling)
                print(
                    f"Warning: Scalers not found for {split} set. Using identity scaling."
                )
                self.scalers["global_mean"] = np.zeros(NUM_GLOBAL_FEATURES)
                self.scalers["global_std"] = np.ones(NUM_GLOBAL_FEATURES)
                self.scalers["target_mean"] = np.zeros(NUM_TARGETS)
                self.scalers["target_std"] = np.ones(NUM_TARGETS)

        # Apply Normalization
        self.global_feats_norm = (
            self.global_feats - self.scalers["global_mean"]
        ) / self.scalers["global_std"]

        if (
            self.targets is not None
            and len(self.targets) > 0
            and self.targets.shape[1] > 0
        ):
            self.targets_norm = (
                self.targets - self.scalers["target_mean"]
            ) / self.scalers["target_std"]
        else:
            self.targets_norm = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Node Features: [num_nodes] (Atomic numbers/indices)
        x = torch.tensor(self.node_feats_list[idx], dtype=torch.long)

        # Edge Index: [2, num_edges]
        edge_index = torch.tensor(self.edge_index_list[idx], dtype=torch.long)

        # Edge Attributes (Distances): [num_edges, 1]
        # We ensure it's 2D for compatibility with GNN layers usually expecting [E, D]
        edge_attr = torch.tensor(self.edge_dist_list[idx], dtype=torch.float).unsqueeze(
            1
        )

        # Global Features: [1, num_global_features]
        global_feat = torch.tensor(
            self.global_feats_norm[idx], dtype=torch.float
        ).unsqueeze(0)

        # Targets
        if self.targets_norm is not None:
            y = torch.tensor(self.targets_norm[idx], dtype=torch.float).unsqueeze(0)
        else:
            # For test set, we might not have targets, but PyG Data object handles None fine,
            # or we can put a placeholder.
            y = None

        # ID
        sample_id = self.ids[idx]

        # Construct Data object
        # Note: We pass 'id' as an attribute. PyG batches attributes that are not standard (x, edge_index, etc.)
        # by simply creating a list/array of them if they don't match node/edge dim.
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_feat=global_feat,
            y=y,
            id=sample_id,
        )

        return data


def get_loaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to load cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, scalers)
    """
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Train Dataset
    print("Initializing Train Dataset...")
    train_dataset = CrystalDataset(
        metadata_path=TRAIN_METADATA_PATH,
        cache_path=TRAIN_CACHE_PATH,
        split="train",
        load_cached_data=load_cached_data,
    )

    # Retrieve scalers computed on training set
    scalers = train_dataset.scalers

    # 2. Validation Dataset
    print("Initializing Validation Dataset...")
    val_dataset = CrystalDataset(
        metadata_path=VAL_METADATA_PATH,
        cache_path=VAL_CACHE_PATH,
        split="val",
        scalers=scalers,
        load_cached_data=load_cached_data,
    )

    # 3. Test Dataset
    print("Initializing Test Dataset...")
    test_dataset = CrystalDataset(
        metadata_path=TEST_METADATA_PATH,
        cache_path=TEST_CACHE_PATH,
        split="test",
        scalers=scalers,
        load_cached_data=load_cached_data,
    )

    # 4. Create DataLoaders
    # PyG DataLoader handles batching of graphs (diagonal stacking of adjacency matrices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scalers
