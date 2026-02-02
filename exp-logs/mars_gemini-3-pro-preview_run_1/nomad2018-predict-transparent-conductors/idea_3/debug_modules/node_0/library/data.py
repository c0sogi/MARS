import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.features import process_dataset


class MaterialsDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_path,
        load_cached_data=True,
        scaler=None,
        is_test=False,
    ):
        """
        PyTorch Dataset for materials data.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            cache_path (str): Path to the .npz cache file.
            load_cached_data (bool): Whether to load data from cache if available.
            scaler (StandardScaler, optional): Scaler for lattice features. If None, fits a new one.
            is_test (bool): Flag indicating if this is the test set (targets may be dummies).
        """
        # Load and process data using the library function (handles caching logic)
        data_dict = process_dataset(
            metadata_path=metadata_path,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
        )

        self.ids = data_dict["ids"]
        self.atomic_features = data_dict[
            "atomic_features"
        ]  # Object array of (N_atoms, D) arrays
        self.lattice_features = data_dict["lattice_features"]  # (N_samples, 7)
        self.targets = data_dict["targets"]  # (N_samples, 2), log-transformed
        self.is_test = is_test

        # Handle standardization of lattice features
        if scaler is None:
            self.scaler = StandardScaler()
            self.lattice_features = self.scaler.fit_transform(self.lattice_features)
        else:
            self.scaler = scaler
            self.lattice_features = self.scaler.transform(self.lattice_features)

        # Convert fixed-size arrays to tensors
        self.lattice_features = torch.tensor(self.lattice_features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Atomic features are ragged, convert to tensor on demand
        atom_feats = torch.tensor(self.atomic_features[idx], dtype=torch.float32)

        return {
            "id": self.ids[idx],
            "atomic_features": atom_feats,
            "lattice_features": self.lattice_features[idx],
            "targets": self.targets[idx],
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms per crystal.

    Constructs a single large tensor for atomic features and a batch index vector
    for pooling operations.
    """
    batch_ids = []
    batch_atomic_features = []
    batch_lattice_features = []
    batch_targets = []
    batch_indices = []

    for i, sample in enumerate(batch):
        batch_ids.append(sample["id"])

        # Handle atomic features (ragged)
        atoms = sample["atomic_features"]
        batch_atomic_features.append(atoms)

        # Create batch index vector: [i, i, ..., i] for each atom in this sample
        num_atoms = atoms.shape[0]
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Handle fixed-size features
        batch_lattice_features.append(sample["lattice_features"])
        batch_targets.append(sample["targets"])

    # Concatenate ragged tensors along the first dimension (atoms)
    if batch_atomic_features:
        batch_atomic_features = torch.cat(batch_atomic_features, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)
    else:
        batch_atomic_features = torch.tensor([], dtype=torch.float32)
        batch_indices = torch.tensor([], dtype=torch.long)

    # Stack fixed-size tensors along the batch dimension
    batch_lattice_features = torch.stack(batch_lattice_features, dim=0)
    batch_targets = torch.stack(batch_targets, dim=0)
    batch_ids = torch.tensor(batch_ids, dtype=torch.long)

    return {
        "ids": batch_ids,
        "atomic_features": batch_atomic_features,
        "batch_indices": batch_indices,
        "lattice_features": batch_lattice_features,
        "targets": batch_targets,
    }


def get_datasets(load_cached_data=True):
    """
    Factory function to initialize Train, Validation, and Test datasets.
    Ensures the scaler fitted on the training set is applied to val and test sets.
    """
    # 1. Train Dataset (Fits the scaler)
    train_dataset = MaterialsDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
        scaler=None,
        is_test=False,
    )

    # 2. Validation Dataset (Uses train scaler)
    val_dataset = MaterialsDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.VAL_CACHE_PATH,
        load_cached_data=load_cached_data,
        scaler=train_dataset.scaler,
        is_test=False,
    )

    # 3. Test Dataset (Uses train scaler)
    test_dataset = MaterialsDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=Config.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
        scaler=train_dataset.scaler,
        is_test=True,
    )

    return train_dataset, val_dataset, test_dataset
