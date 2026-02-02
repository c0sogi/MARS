import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import CONFIG
from library.features import process_dataset


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures.
    Stores atomic features, global features, targets, and IDs.
    """

    def __init__(self, atomic_feats, global_feats, targets, ids):
        self.atomic_feats = atomic_feats
        self.global_feats = global_feats
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.global_feats)

    def __getitem__(self, idx):
        # atomic_feats is a list of arrays (one array per crystal)
        return (
            torch.tensor(self.atomic_feats[idx], dtype=torch.float32),
            torch.tensor(self.global_feats[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            self.ids[idx],
        )


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms per crystal.
    Concatenates atomic features and creates a batch index tensor.
    """
    atomic_list, global_list, target_list, id_list = zip(*batch)

    # Create batch indices for atoms (to know which atom belongs to which crystal)
    batch_indices = []
    for i, atoms in enumerate(atomic_list):
        batch_indices.append(torch.full((atoms.shape[0],), i, dtype=torch.long))

    # Concatenate all atoms into a single large tensor
    atomic_batch = torch.cat(atomic_list, dim=0)
    batch_indices = torch.cat(batch_indices, dim=0)

    # Stack global features and targets (fixed size per crystal)
    global_batch = torch.stack(global_list, dim=0)
    target_batch = torch.stack(target_list, dim=0)

    return atomic_batch, batch_indices, global_batch, target_batch, id_list


def get_data_loaders(
    input_dir="./input", batch_size=CONFIG["batch_size"], load_cached_data=True
):
    """
    Orchestrates the data pipeline: loading, processing, scaling, and batching.

    Args:
        input_dir (str): Path to input directory.
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler_atomic, scaler_global)
    """
    # 1. Load Metadata
    # Using the pre-split metadata files
    train_df = pd.read_csv(os.path.join("./metadata", "train.csv"))
    val_df = pd.read_csv(os.path.join("./metadata", "val.csv"))
    test_df = pd.read_csv(os.path.join("./metadata", "test.csv"))

    # 2. Process Data (Feature Extraction)
    # Leverages library.features.process_dataset which handles geometry parsing and caching
    print("Processing Train Data...")
    train_atomic, train_global, train_targets, train_ids = process_dataset(
        train_df, input_dir, load_cached_data=load_cached_data, cache_name="train"
    )

    print("Processing Val Data...")
    val_atomic, val_global, val_targets, val_ids = process_dataset(
        val_df, input_dir, load_cached_data=load_cached_data, cache_name="val"
    )

    print("Processing Test Data...")
    test_atomic, test_global, test_targets, test_ids = process_dataset(
        test_df, input_dir, load_cached_data=load_cached_data, cache_name="test"
    )

    # 3. Feature Scaling
    print("Fitting Scalers...")

    # Atomic Features: Stack all training atoms to fit the scaler
    # train_atomic is a list of (N_atoms, n_features) arrays
    all_train_atomic = np.vstack(train_atomic)
    scaler_atomic = StandardScaler()
    scaler_atomic.fit(all_train_atomic)

    # Global Features: Standard scaling
    scaler_global = StandardScaler()
    scaler_global.fit(train_global)

    # Apply Scaling
    def scale_atomic_list(atomic_list, scaler):
        return [scaler.transform(x) for x in atomic_list]

    train_atomic_scaled = scale_atomic_list(train_atomic, scaler_atomic)
    val_atomic_scaled = scale_atomic_list(val_atomic, scaler_atomic)
    test_atomic_scaled = scale_atomic_list(test_atomic, scaler_atomic)

    train_global_scaled = scaler_global.transform(train_global)
    val_global_scaled = scaler_global.transform(val_global)
    test_global_scaled = scaler_global.transform(test_global)

    # 4. Target Transformation
    # Apply log(1+x) to targets to handle skewness and ensure positivity
    train_targets_log = np.log1p(train_targets)
    val_targets_log = np.log1p(val_targets)
    # Test targets are placeholders, no transformation needed for inference logic

    # 5. Create Datasets
    train_dataset = CrystalDataset(
        train_atomic_scaled, train_global_scaled, train_targets_log, train_ids
    )
    val_dataset = CrystalDataset(
        val_atomic_scaled, val_global_scaled, val_targets_log, val_ids
    )
    test_dataset = CrystalDataset(
        test_atomic_scaled, test_global_scaled, test_targets, test_ids
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler_atomic, scaler_global
