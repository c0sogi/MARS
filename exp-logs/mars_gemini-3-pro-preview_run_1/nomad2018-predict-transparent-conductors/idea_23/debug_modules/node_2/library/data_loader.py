import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.feature_extractor import process_dataset
import library.config as config


class CrystalDataset(Dataset):
    """
    Dataset class for crystal structures.
    Loads processed features (atomic and global) and targets.
    """

    def __init__(
        self,
        metadata_path,
        cache_path,
        load_cached_data=True,
        fit_scalers=False,
        max_samples=None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            cache_path (str): Path to the .npz cache file.
            load_cached_data (bool): Whether to load from cache if available.
            fit_scalers (bool): Whether to fit scalers (True for training set).
            max_samples (int, optional): Limit the number of samples for debugging.
        """
        self.data = process_dataset(
            metadata_path=metadata_path,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
            fit_scalers=fit_scalers,
            scalers_path=config.SCALERS_CACHE_PATH,
        )

        self.atomic_features = self.data["atomic_features"]
        self.global_features = self.data["global_features"]
        self.targets = self.data["targets"]
        self.ids = self.data["ids"]

        if max_samples is not None:
            self.atomic_features = self.atomic_features[:max_samples]
            self.global_features = self.global_features[:max_samples]
            self.targets = self.targets[:max_samples]
            self.ids = self.ids[:max_samples]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Atomic features: (N_atoms, 12)
        # Convert numpy array to torch tensor
        atomic_feat = torch.tensor(self.atomic_features[idx], dtype=torch.float32)

        # Global features: (12,)
        global_feat = torch.tensor(self.global_features[idx], dtype=torch.float32)

        # Targets: (2,) - formation_energy, bandgap_energy
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        # ID
        sample_id = self.ids[idx]

        return atomic_feat, global_feat, target, sample_id


def collate_batch(batch):
    """
    Collates a batch of variable-sized crystal graphs into flat tensors.

    Args:
        batch: List of tuples (atomic_feat, global_feat, target, sample_id)

    Returns:
        batch_atomic: (Sum_N_atoms, 12)
        batch_indices: (Sum_N_atoms,) indicating which sample each atom belongs to
        batch_global: (B, 12)
        batch_targets: (B, 2)
        batch_ids: (B,)
    """
    atomic_feats_list = []
    batch_indices_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    for i, (atomic_feat, global_feat, target, sample_id) in enumerate(batch):
        n_atoms = atomic_feat.shape[0]

        atomic_feats_list.append(atomic_feat)
        # Create index vector [i, i, ..., i] for the current sample
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        global_feats_list.append(global_feat)
        targets_list.append(target)
        ids_list.append(sample_id)

    # Concatenate all atomic features into one large tensor
    batch_atomic = torch.cat(atomic_feats_list, dim=0)

    # Concatenate all batch indices
    batch_indices = torch.cat(batch_indices_list, dim=0)

    # Stack global features and targets
    batch_global = torch.stack(global_feats_list, dim=0)
    batch_targets = torch.stack(targets_list, dim=0)

    # Convert IDs to tensor
    batch_ids = torch.tensor(ids_list, dtype=torch.int32)

    return batch_atomic, batch_indices, batch_global, batch_targets, batch_ids


def get_dataloaders(
    batch_size=None, num_workers=0, load_cached_data=True, max_samples=None
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use cached data.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Cite debug_lesson_14: Default Arguments Bind at Definition Time
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Train Dataset (Fits scalers)
    train_dataset = CrystalDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        cache_path=config.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
        fit_scalers=True,
        max_samples=max_samples,
    )

    # Validation Dataset (Uses scalers fitted on train)
    val_dataset = CrystalDataset(
        metadata_path=config.VAL_METADATA_PATH,
        cache_path=config.VAL_CACHE_PATH,
        load_cached_data=load_cached_data,
        fit_scalers=False,
        max_samples=max_samples,
    )

    # Test Dataset (Uses scalers fitted on train)
    test_dataset = CrystalDataset(
        metadata_path=config.TEST_METADATA_PATH,
        cache_path=config.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
        fit_scalers=False,
        max_samples=max_samples,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
