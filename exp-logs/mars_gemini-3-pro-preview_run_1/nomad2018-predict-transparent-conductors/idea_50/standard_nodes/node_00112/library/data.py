import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.geometry import process_dataset
from library.utils import SelectiveScaler


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material crystals.
    Handles the mapping from stacked numpy arrays (from process_dataset)
    to individual crystal samples.
    """

    def __init__(self, atomic_features, global_features, targets, ids, batch_indices):
        """
        Args:
            atomic_features (np.ndarray): (Total_Atoms, Atomic_Dim)
            global_features (np.ndarray): (Num_Crystals, Global_Dim)
            targets (np.ndarray): (Num_Crystals, 2)
            ids (np.ndarray): (Num_Crystals,)
            batch_indices (np.ndarray): (Total_Atoms,) mapping atoms to crystal index 0..N-1
        """
        # We need to split the huge atomic_features array into a list of arrays,
        # one per crystal, to make __getitem__ efficient.
        # batch_indices is guaranteed to be sorted and contiguous (0,0,..,1,1,..,N-1)
        # by the geometry.process_dataset function.

        # Get the number of atoms for each crystal
        _, counts = np.unique(batch_indices, return_counts=True)

        # Calculate split points
        split_points = np.cumsum(counts)[:-1]

        # Split into list of arrays
        self.atomic_features_list = np.split(atomic_features, split_points)

        self.global_features = global_features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.ids = ids

        # Sanity check
        assert (
            len(self.atomic_features_list)
            == len(self.global_features)
            == len(self.targets)
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns:
            atomic_feats (Tensor): (N_atoms, Atomic_Dim)
            global_feats (Tensor): (Global_Dim,)
            target (Tensor): (2,)
            id (int): Crystal ID
        """
        return (
            torch.from_numpy(self.atomic_features_list[idx]),
            torch.from_numpy(self.global_features[idx]),
            torch.from_numpy(self.targets[idx]),
            self.ids[idx],
        )


def sparse_collate_fn(batch):
    """
    Custom collate function for sparse batching.
    Concatenates atomic features from multiple crystals and creates a batch_index vector.

    Args:
        batch: List of tuples (atomic, global, target, id)

    Returns:
        batch_atomic (Tensor): (Total_Batch_Atoms, Atomic_Dim)
        batch_global (Tensor): (Batch_Size, Global_Dim)
        batch_indices (Tensor): (Total_Batch_Atoms,) mapping atoms to batch sample index
        batch_targets (Tensor): (Batch_Size, 2)
        batch_ids (Tensor): (Batch_Size,)
    """
    atomic_list, global_list, target_list, id_list = zip(*batch)

    # 1. Concatenate atomic features into one big tensor
    batch_atomic = torch.cat(atomic_list, dim=0)

    # 2. Create batch indices vector (0,0,0, 1,1, 2,2,2,2 ...)
    sizes = [a.shape[0] for a in atomic_list]
    batch_indices = []
    for i, size in enumerate(sizes):
        batch_indices.append(torch.full((size,), i, dtype=torch.long))
    batch_indices = torch.cat(batch_indices, dim=0)

    # 3. Stack global features
    batch_global = torch.stack(global_list, dim=0)

    # 4. Stack targets
    batch_targets = torch.stack(target_list, dim=0)

    # 5. Stack IDs
    batch_ids = torch.tensor(id_list, dtype=torch.long)

    return batch_atomic, batch_global, batch_indices, batch_targets, batch_ids


def get_train_val_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.
    Handles processing, scaling, and caching.
    """
    # 1. Process Data (Load from cache or compute)
    train_data = process_dataset(
        Config.TRAIN_META_PATH,
        load_cached_data=load_cached_data,
        cache_name="train_data",
    )
    val_data = process_dataset(
        Config.VAL_META_PATH, load_cached_data=load_cached_data, cache_name="val_data"
    )

    # 2. Define Scaling Columns
    # Atomic Features: First 4 are One-Hot (Al, Ga, In, O), do not scale.
    # The rest (coords, distances, ratios, contexts) are continuous.
    atomic_scale_cols = list(range(4, Config.ATOMIC_FEATURE_DIM))

    # Global Features: All are continuous physical properties.
    global_scale_cols = list(range(Config.GLOBAL_FEATURE_DIM))

    # 3. Initialize and Fit Scalers on Training Data
    atomic_scaler = SelectiveScaler(cols_to_scale=atomic_scale_cols)
    global_scaler = SelectiveScaler(cols_to_scale=global_scale_cols)

    atomic_scaler.fit(train_data["atomic_features"])
    global_scaler.fit(train_data["global_features"])

    # 4. Save Scalers for Inference
    atomic_scaler.save(os.path.join(Config.EXECUTION_DIR, "scalers_atomic.npz"))
    global_scaler.save(os.path.join(Config.EXECUTION_DIR, "scalers_global.npz"))

    # 5. Transform Data
    train_data["atomic_features"] = atomic_scaler.transform(
        train_data["atomic_features"]
    )
    train_data["global_features"] = global_scaler.transform(
        train_data["global_features"]
    )

    val_data["atomic_features"] = atomic_scaler.transform(val_data["atomic_features"])
    val_data["global_features"] = global_scaler.transform(val_data["global_features"])

    # 6. Create Datasets
    train_dataset = MaterialDataset(
        train_data["atomic_features"],
        train_data["global_features"],
        train_data["targets"],
        train_data["ids"],
        train_data["batch_indices"],
    )

    val_dataset = MaterialDataset(
        val_data["atomic_features"],
        val_data["global_features"],
        val_data["targets"],
        val_data["ids"],
        val_data["batch_indices"],
    )

    # 7. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=sparse_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Prepares DataLoader for testing/inference.
    Uses scalers fitted on training data.
    """
    # 1. Process Data
    test_data = process_dataset(
        Config.TEST_META_PATH, load_cached_data=load_cached_data, cache_name="test_data"
    )

    # 2. Load Scalers
    atomic_scaler_path = os.path.join(Config.EXECUTION_DIR, "scalers_atomic.npz")
    global_scaler_path = os.path.join(Config.EXECUTION_DIR, "scalers_global.npz")

    if not os.path.exists(atomic_scaler_path) or not os.path.exists(global_scaler_path):
        raise FileNotFoundError("Scalers not found. Run training first.")

    atomic_scaler = SelectiveScaler().load(atomic_scaler_path)
    global_scaler = SelectiveScaler().load(global_scaler_path)

    # 3. Transform Data
    test_data["atomic_features"] = atomic_scaler.transform(test_data["atomic_features"])
    test_data["global_features"] = global_scaler.transform(test_data["global_features"])

    # 4. Create Dataset
    # Note: Targets in test_data are placeholders [0.0, 0.0]
    test_dataset = MaterialDataset(
        test_data["atomic_features"],
        test_data["global_features"],
        test_data["targets"],
        test_data["ids"],
        test_data["batch_indices"],
    )

    # 5. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=sparse_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
