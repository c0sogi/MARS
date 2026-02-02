import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SCALERS_CACHE_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.geometry import load_and_process_data
from library.utils import read_metadata


class TargetTransformer:
    """
    Log-transforms targets to align MSE loss with RMSLE metric.
    y -> log(1 + y)
    """

    def transform(self, y):
        return torch.log1p(y)

    def inverse_transform(self, y):
        return torch.expm1(y)


class FeatureScaler:
    """
    Standardizes features by removing mean and scaling to unit variance.
    Handles atomic features (selective scaling) and global features.
    """

    def __init__(self):
        self.atomic_mean = None
        self.atomic_std = None
        self.global_mean = None
        self.global_std = None
        self.fit_done = False

    def fit(self, atomic_features_list, global_features_tensor):
        """
        Compute mean and std from training data.
        atomic_features_list: list of tensors (N_i, 9)
        global_features_tensor: tensor (B, 12)
        """
        # 1. Atomic Features
        # Concatenate all atoms to compute global stats for the atomic features
        all_atomic = torch.cat(atomic_features_list, dim=0)

        # Columns 0-3 are One-Hot (Al, Ga, In, O) -> Do not scale
        # Columns 4-6 are Centered Coords -> Scale
        # Columns 7-8 are Homo/Hetero Distances -> Scale
        features_to_scale = all_atomic[:, 4:]

        self.atomic_mean = features_to_scale.mean(dim=0)
        self.atomic_std = features_to_scale.std(dim=0)

        # Prevent division by zero
        self.atomic_std[self.atomic_std == 0] = 1.0

        # 2. Global Features
        # Scale all 12 dimensions
        self.global_mean = global_features_tensor.mean(dim=0)
        self.global_std = global_features_tensor.std(dim=0)
        self.global_std[self.global_std == 0] = 1.0

        self.fit_done = True

    def transform_atomic(self, atomic_tensor):
        """
        Apply scaling to atomic features (cols 4:9).
        atomic_tensor: (N, 9)
        """
        if not self.fit_done:
            raise RuntimeError("Scaler not fitted")

        # Clone to avoid in-place modification of cached data
        out = atomic_tensor.clone()
        out[:, 4:] = (out[:, 4:] - self.atomic_mean) / self.atomic_std
        return out

    def transform_global(self, global_tensor):
        """
        Apply scaling to global features.
        global_tensor: (12,)
        """
        if not self.fit_done:
            raise RuntimeError("Scaler not fitted")
        return (global_tensor - self.global_mean) / self.global_std

    def save(self, path):
        state = {
            "atomic_mean": self.atomic_mean,
            "atomic_std": self.atomic_std,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.atomic_mean = state["atomic_mean"]
        self.atomic_std = state["atomic_std"]
        self.global_mean = state["global_mean"]
        self.global_std = state["global_std"]
        self.fit_done = True


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for Crystal structures.
    """

    def __init__(self, data_dict, scaler=None, transform_target=False):
        """
        Args:
            data_dict (dict): Dictionary containing processed data lists/tensors.
            scaler (FeatureScaler): Fitted scaler instance.
            transform_target (bool): Whether to log-transform targets.
        """
        self.atomic_features = data_dict["atomic_features"]  # List of (N, 9)
        self.global_features = data_dict["global_features"]  # Tensor (B, 12)
        self.targets = data_dict["targets"]  # Tensor (B, 2)
        self.ids = data_dict["ids"]  # List of ints

        self.scaler = scaler
        self.target_transformer = TargetTransformer() if transform_target else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        atomic = self.atomic_features[idx]
        glob = self.global_features[idx]
        target = self.targets[idx]
        id_val = self.ids[idx]

        # Apply Feature Scaling
        if self.scaler:
            atomic = self.scaler.transform_atomic(atomic)
            glob = self.scaler.transform_global(glob)

        # Apply Target Transformation
        if self.target_transformer:
            target = self.target_transformer.transform(target)

        return atomic, glob, target, id_val


class CrystalCollate:
    """
    Collate function to batch variable-size point clouds.
    Creates a batch index vector for pooling operations.
    """

    def __call__(self, batch):
        # batch is a list of tuples: (atomic, global, target, id)
        atomic_list, global_list, target_list, id_list = zip(*batch)

        # 1. Create batch index vector
        # e.g. [0, 0, 0, 1, 1, 2, 2, 2, 2]
        batch_indices = []
        for i, atom_tensor in enumerate(atomic_list):
            n_atoms = atom_tensor.shape[0]
            batch_indices.append(torch.full((n_atoms,), i, dtype=torch.long))

        batch_index = torch.cat(batch_indices)

        # 2. Concatenate atomic features into one large tensor (Total_Atoms, 9)
        atomic_batch = torch.cat(atomic_list, dim=0)

        # 3. Stack global features (B, 12)
        global_batch = torch.stack(global_list)

        # 4. Stack targets (B, 2)
        target_batch = torch.stack(target_list)

        return atomic_batch, batch_index, global_batch, target_batch, list(id_list)


def get_dataloaders(debug_sample_size=None):
    """
    Orchestrates data loading, processing, scaling, and DataLoader creation.

    Args:
        debug_sample_size (int): If set, limits the number of samples for debugging.

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # 1. Load Metadata
    train_meta = read_metadata(TRAIN_METADATA_PATH)
    val_meta = read_metadata(VAL_METADATA_PATH)
    test_meta = read_metadata(TEST_METADATA_PATH)

    if debug_sample_size:
        print(f"DEBUG MODE: Limiting datasets to {debug_sample_size} samples.")
        train_meta = train_meta.head(debug_sample_size)
        val_meta = val_meta.head(debug_sample_size)
        test_meta = test_meta.head(debug_sample_size)

    # 2. Process Data (Geometry parsing + Feature Engineering)
    # This uses the library function which handles caching
    print("Preparing Training Data...")
    train_data = load_and_process_data(train_meta, TRAIN_CACHE_PATH)

    print("Preparing Validation Data...")
    val_data = load_and_process_data(val_meta, VAL_CACHE_PATH)

    print("Preparing Test Data...")
    test_data = load_and_process_data(test_meta, TEST_CACHE_PATH)

    # 3. Fit or Load Scaler
    scaler = FeatureScaler()
    if os.path.exists(SCALERS_CACHE_PATH):
        print(f"Loading scalers from {SCALERS_CACHE_PATH}...")
        scaler.load(SCALERS_CACHE_PATH)
    else:
        print("Fitting scalers on training data...")
        scaler.fit(train_data["atomic_features"], train_data["global_features"])
        scaler.save(SCALERS_CACHE_PATH)

    # 4. Create Datasets
    # Train/Val targets are log-transformed
    train_dataset = CrystalDataset(train_data, scaler=scaler, transform_target=True)
    val_dataset = CrystalDataset(val_data, scaler=scaler, transform_target=True)
    # Test targets are not transformed (they are placeholders/ignored)
    test_dataset = CrystalDataset(test_data, scaler=scaler, transform_target=False)

    # 5. Create DataLoaders
    collate_fn = CrystalCollate()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler
