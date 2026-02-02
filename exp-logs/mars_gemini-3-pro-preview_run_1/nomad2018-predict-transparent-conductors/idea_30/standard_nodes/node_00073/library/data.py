import os
import numpy as np
import torch
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
    SEED,
)
from library.features import process_dataset
from library.utils import log_transform_targets


class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for material crystals.
    Handles loading, scaling, and target transformation.
    """

    def __init__(self, metadata_path, cache_path, split="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            cache_path (str): Path to the .npz cache file for this split.
            split (str): 'train', 'val', or 'test'. Determines scaling behavior.
            load_cached_data (bool): Whether to load from cache if available.
        """
        self.split = split

        # 1. Load Data (uses library.features caching mechanism)
        data_dict = process_dataset(
            metadata_path, cache_path, load_cached_data=load_cached_data
        )

        self.atomic_features_flat = data_dict["atomic_features_flat"]
        self.atom_counts = data_dict["atom_counts"]
        self.global_features = data_dict["global_features"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

        self.n_samples = len(self.ids)

        # Pre-calculate start indices for slicing atomic features
        self.atom_indices = np.concatenate(([0], np.cumsum(self.atom_counts)))

        # 2. Handle Scaling
        self._apply_scaling()

        # 3. Transform Targets (Log transform for energy values)
        # We transform targets for train and val to train/evaluate on RMSLE-aligned loss.
        if self.split in ["train", "val"]:
            self.targets = log_transform_targets(self.targets)

    def _apply_scaling(self):
        """
        Applies standard scaling to continuous features.
        Fits on 'train', transforms on 'val'/'test'.
        """
        # Atomic features: 0-3 are one-hot (skip), 4-8 are continuous (scale)
        ATOMIC_CONT_START = 4

        if self.split == "train":
            # Compute statistics
            atomic_mean = np.mean(
                self.atomic_features_flat[:, ATOMIC_CONT_START:], axis=0
            )
            atomic_std = np.std(
                self.atomic_features_flat[:, ATOMIC_CONT_START:], axis=0
            )
            # Prevent division by zero
            atomic_std[atomic_std == 0] = 1.0

            global_mean = np.mean(self.global_features, axis=0)
            global_std = np.std(self.global_features, axis=0)
            global_std[global_std == 0] = 1.0

            # Save scalers
            np.savez(
                SCALERS_CACHE_PATH,
                atomic_mean=atomic_mean,
                atomic_std=atomic_std,
                global_mean=global_mean,
                global_std=global_std,
            )
        else:
            # Load scalers
            if not os.path.exists(SCALERS_CACHE_PATH):
                raise FileNotFoundError(
                    f"Scalers not found at {SCALERS_CACHE_PATH}. Run training set first."
                )

            scalers = np.load(SCALERS_CACHE_PATH)
            atomic_mean = scalers["atomic_mean"]
            atomic_std = scalers["atomic_std"]
            global_mean = scalers["global_mean"]
            global_std = scalers["global_std"]

        # Apply scaling (Standardization)
        self.atomic_features_flat[:, ATOMIC_CONT_START:] = (
            self.atomic_features_flat[:, ATOMIC_CONT_START:] - atomic_mean
        ) / atomic_std

        self.global_features = (self.global_features - global_mean) / global_std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Slice atomic features for the specific crystal
        start = self.atom_indices[idx]
        end = self.atom_indices[idx + 1]

        atomic_feats = torch.tensor(
            self.atomic_features_flat[start:end], dtype=torch.float32
        )
        global_feats = torch.tensor(self.global_features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        sample_id = self.ids[idx]

        return atomic_feats, global_feats, target, sample_id


def collate_crystals(batch):
    """
    Collate function for the DataLoader.
    Batches crystals by concatenating atomic features and creating a batch index vector.

    Args:
        batch: List of tuples (atomic_feats, global_feats, target, sample_id)

    Returns:
        batch_atomic_feats: (Total_Atoms_In_Batch, Atomic_Dim)
        batch_indices: (Total_Atoms_In_Batch,) - maps atoms to their crystal index in batch
        batch_global_feats: (Batch_Size, Global_Dim)
        batch_targets: (Batch_Size, Target_Dim)
        batch_ids: (Batch_Size,)
    """
    atomic_feats_list = []
    batch_indices_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    for i, (atomic_f, global_f, target, sample_id) in enumerate(batch):
        n_atoms = atomic_f.shape[0]

        atomic_feats_list.append(atomic_f)
        # Create an index vector [i, i, ..., i] for the current crystal
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        global_feats_list.append(global_f)
        targets_list.append(target)
        ids_list.append(sample_id)

    # Concatenate all
    batch_atomic_feats = torch.cat(atomic_feats_list, dim=0)
    batch_indices = torch.cat(batch_indices_list, dim=0)
    batch_global_feats = torch.stack(global_feats_list, dim=0)
    batch_targets = torch.stack(targets_list, dim=0)
    batch_ids = torch.tensor(ids_list, dtype=torch.int32)

    return (
        batch_atomic_feats,
        batch_indices,
        batch_global_feats,
        batch_targets,
        batch_ids,
    )


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """
    # Create Datasets
    # Note: Train dataset must be initialized first to generate/save scalers
    train_dataset = MaterialsDataset(
        TRAIN_METADATA_PATH,
        TRAIN_CACHE_PATH,
        split="train",
        load_cached_data=load_cached_data,
    )

    val_dataset = MaterialsDataset(
        VAL_METADATA_PATH,
        VAL_CACHE_PATH,
        split="val",
        load_cached_data=load_cached_data,
    )

    test_dataset = MaterialsDataset(
        TEST_METADATA_PATH,
        TEST_CACHE_PATH,
        split="test",
        load_cached_data=load_cached_data,
    )

    # Create DataLoaders
    # Use a fixed generator for reproducibility
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_crystals,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id),
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
