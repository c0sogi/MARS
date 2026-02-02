import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from library.config import Config
from library.geometry import GeometryProcessor, process_dataset
from library.utils import StandardScaler, log_transform


class MaterialDataset(Dataset):
    def __init__(self, data_dict):
        """
        PyTorch Dataset for material data.

        Args:
            data_dict: Dictionary containing processed data from process_dataset.
                       Expected keys: 'atomic_features', 'global_features',
                       'batch_indices', 'targets', 'ids'.
        """
        self.atomic_features = data_dict["atomic_features"]
        self.global_features = data_dict["global_features"]
        self.batch_indices = data_dict["batch_indices"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Determine the slice range for the atoms belonging to this crystal
        start_idx = self.batch_indices[idx]
        end_idx = self.batch_indices[idx + 1]

        # Slice the flattened atomic features array
        atom_feats = self.atomic_features[start_idx:end_idx]

        # Retrieve global features, target, and ID for this crystal
        glob_feats = self.global_features[idx]
        target = self.targets[idx]
        id_val = self.ids[idx]

        # Return tensors
        return {
            "atomic_features": torch.tensor(atom_feats, dtype=torch.float32),
            "global_features": torch.tensor(glob_feats, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "id": torch.tensor([id_val], dtype=torch.long),
        }


def collate_batch(batch):
    """
    Custom collate function to handle variable number of atoms per crystal.

    Args:
        batch: List of dictionaries returned by __getitem__.

    Returns:
        Dictionary containing batched tensors. 'atomic_features' are concatenated
        along the first dimension. 'batch_index' maps each atom to its sample index.
    """
    batch_atomic_features = []
    batch_global_features = []
    batch_targets = []
    batch_ids = []
    batch_indices = []

    for i, sample in enumerate(batch):
        atom_feats = sample["atomic_features"]
        num_atoms = atom_feats.shape[0]

        batch_atomic_features.append(atom_feats)
        batch_global_features.append(sample["global_features"])
        batch_targets.append(sample["target"])
        batch_ids.append(sample["id"])

        # Create batch index vector (e.g., [0, 0, ..., 1, 1, ...])
        # This is used for scatter_mean/scatter_max pooling in the model
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))

    return {
        "atomic_features": torch.cat(batch_atomic_features, dim=0),
        "global_features": torch.stack(batch_global_features, dim=0),
        "batch_index": torch.cat(batch_indices, dim=0),
        "targets": torch.stack(batch_targets, dim=0),
        "ids": torch.cat(batch_ids, dim=0),
    }


def get_train_val_loaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.

    1. Processes raw geometry files (with caching).
    2. Fits StandardScalers on the training set.
    3. Transforms features and log-transforms targets.
    4. Saves scalers for inference.

    Args:
        batch_size: Batch size for DataLoaders.
        load_cached_data: Whether to try loading processed data from disk.

    Returns:
        train_loader, val_loader
    """
    processor = GeometryProcessor()

    # Process Training Data
    print("Processing Training Data...")
    train_data = process_dataset(
        Config.TRAIN_METADATA,
        processor,
        load_cached_data=load_cached_data,
        cache_path=Config.TRAIN_CACHE,
    )

    # Process Validation Data
    print("Processing Validation Data...")
    val_data = process_dataset(
        Config.VAL_METADATA,
        processor,
        load_cached_data=load_cached_data,
        cache_path=Config.VAL_CACHE,
    )

    # Initialize Scalers
    atomic_scaler = StandardScaler()
    global_scaler = StandardScaler()

    # Fit scalers on training data only
    print("Fitting scalers on training data...")
    atomic_scaler.fit(train_data["atomic_features"])
    global_scaler.fit(train_data["global_features"])

    # Save scalers for use during testing/inference
    atomic_scaler_path = Config.SCALERS_CACHE.replace(".npz", "_atomic.npz")
    global_scaler_path = Config.SCALERS_CACHE.replace(".npz", "_global.npz")
    print(f"Saving scalers to {atomic_scaler_path} and {global_scaler_path}...")
    atomic_scaler.save(atomic_scaler_path)
    global_scaler.save(global_scaler_path)

    # Transform Data (In-place to save memory)
    print("Transforming features and targets...")
    train_data["atomic_features"] = atomic_scaler.transform(
        train_data["atomic_features"]
    )
    train_data["global_features"] = global_scaler.transform(
        train_data["global_features"]
    )
    # Apply log(1+y) transform to targets
    train_data["targets"] = log_transform(train_data["targets"])

    val_data["atomic_features"] = atomic_scaler.transform(val_data["atomic_features"])
    val_data["global_features"] = global_scaler.transform(val_data["global_features"])
    val_data["targets"] = log_transform(val_data["targets"])

    # Create Datasets
    train_dataset = MaterialDataset(train_data)
    val_dataset = MaterialDataset(val_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Prepares DataLoader for the test set.

    1. Processes raw geometry files (with caching).
    2. Loads pre-fitted scalers.
    3. Transforms features.

    Args:
        batch_size: Batch size for DataLoader.
        load_cached_data: Whether to try loading processed data from disk.

    Returns:
        test_loader
    """
    processor = GeometryProcessor()

    # Process Test Data
    print("Processing Test Data...")
    test_data = process_dataset(
        Config.TEST_METADATA,
        processor,
        load_cached_data=load_cached_data,
        cache_path=Config.TEST_CACHE,
    )

    # Load Scalers
    print("Loading scalers...")
    atomic_scaler = StandardScaler()
    global_scaler = StandardScaler()

    atomic_scaler_path = Config.SCALERS_CACHE.replace(".npz", "_atomic.npz")
    global_scaler_path = Config.SCALERS_CACHE.replace(".npz", "_global.npz")

    if not os.path.exists(atomic_scaler_path) or not os.path.exists(global_scaler_path):
        raise FileNotFoundError(
            "Scalers not found. You must run training first to generate scalers."
        )

    atomic_scaler.load(atomic_scaler_path)
    global_scaler.load(global_scaler_path)

    # Transform Data
    print("Transforming test data...")
    test_data["atomic_features"] = atomic_scaler.transform(test_data["atomic_features"])
    test_data["global_features"] = global_scaler.transform(test_data["global_features"])
    # Note: Targets in test_data are dummy values and should be ignored.

    # Create Dataset
    test_dataset = MaterialDataset(test_data)

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return test_loader
