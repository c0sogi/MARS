import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CACHE_TRAIN_DATA,
    CACHE_VAL_DATA,
    CACHE_TEST_DATA,
    BATCH_SIZE,
    SEED,
    WORKING_DIR,
)
from library.features import process_dataset

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material Science data.
    Wraps pre-processed numpy arrays for atomic features, global features, and targets.
    """

    def __init__(self, atomic_features, global_features, targets=None, ids=None):
        self.atomic_features = atomic_features
        self.global_features = global_features
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # atomic_features[idx] is a numpy array (N_atoms, Atomic_Dim)
        # global_features[idx] is a numpy array (Global_Dim,)

        af = torch.tensor(self.atomic_features[idx], dtype=torch.float32)
        gf = torch.tensor(self.global_features[idx], dtype=torch.float32)

        target = None
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        id_val = self.ids[idx]

        return af, gf, target, id_val


def collate_fn(batch):
    """
    Custom collate function for Sparse (Flattened) Batching.

    Args:
        batch: List of tuples (atomic_features, global_features, target, id)

    Returns:
        batch_atomic: (Total_Atoms_In_Batch, Atomic_Dim)
        batch_index: (Total_Atoms_In_Batch,) - Indices mapping atoms to their crystal in the batch
        batch_global: (Batch_Size, Global_Dim)
        batch_targets: (Batch_Size, Num_Targets) or None
        batch_ids: (Batch_Size,)
    """
    atomic_list = []
    batch_index_list = []
    global_list = []
    target_list = []
    id_list = []

    for i, (af, gf, target, id_val) in enumerate(batch):
        # Atomic Stream: Concatenate variable number of atoms
        n_atoms = af.shape[0]
        atomic_list.append(af)

        # Create batch index vector for this crystal (e.g., [0, 0, ..., 1, 1, ...])
        # This allows the model to aggregate atoms back to their crystal later
        batch_index_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        # Global Stream: Standard stacking
        global_list.append(gf)

        # Targets
        if target is not None:
            target_list.append(target)

        # IDs
        id_list.append(id_val)

    # Concatenate sparse atomic features into one large tensor
    batch_atomic = torch.cat(atomic_list, dim=0)
    batch_index = torch.cat(batch_index_list, dim=0)

    # Stack dense global features
    batch_global = torch.stack(global_list, dim=0)

    # Stack targets if they exist
    batch_targets = None
    if len(target_list) > 0:
        batch_targets = torch.stack(target_list, dim=0)

    batch_ids = torch.tensor(id_list, dtype=torch.long)

    return batch_atomic, batch_index, batch_global, batch_targets, batch_ids


def get_loaders(batch_size=BATCH_SIZE, debug_mode=False, load_cached_data=True):
    """
    Orchestrates data loading, processing, and DataLoader creation.

    Args:
        batch_size (int): Batch size for training/inference.
        debug_mode (bool): If True, uses a small subset of data for quick testing.
        load_cached_data (bool): Whether to attempt loading from cache files.

    Returns:
        train_loader, val_loader, test_loader
    """

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Process/Load Train Data
    print("Processing Train Data...")
    train_af, train_gf, train_y, train_ids = process_dataset(
        TRAIN_CSV, CACHE_TRAIN_DATA, load_cached_data=load_cached_data
    )

    # 2. Process/Load Validation Data
    print("Processing Validation Data...")
    val_af, val_gf, val_y, val_ids = process_dataset(
        VAL_CSV, CACHE_VAL_DATA, load_cached_data=load_cached_data
    )

    # 3. Process/Load Test Data
    print("Processing Test Data...")
    test_af, test_gf, test_y, test_ids = process_dataset(
        TEST_CSV, CACHE_TEST_DATA, load_cached_data=load_cached_data
    )

    # Debug Mode: Slice data to a small subset
    if debug_mode:
        print("Debug Mode: Using 50 samples for each split.")
        limit = 50
        train_af, train_gf, train_y, train_ids = (
            train_af[:limit],
            train_gf[:limit],
            train_y[:limit],
            train_ids[:limit],
        )
        val_af, val_gf, val_y, val_ids = (
            val_af[:limit],
            val_gf[:limit],
            val_y[:limit],
            val_ids[:limit],
        )
        test_af, test_gf, test_ids = test_af[:limit], test_gf[:limit], test_ids[:limit]
        if test_y is not None:
            test_y = test_y[:limit]

    # Create PyTorch Datasets
    train_dataset = MaterialDataset(train_af, train_gf, train_y, train_ids)
    val_dataset = MaterialDataset(val_af, val_gf, val_y, val_ids)
    test_dataset = MaterialDataset(test_af, test_gf, test_y, test_ids)

    # Create DataLoaders
    # num_workers=0 is safer for simple scripts to avoid multiprocessing overhead/issues
    # pin_memory=True speeds up transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
