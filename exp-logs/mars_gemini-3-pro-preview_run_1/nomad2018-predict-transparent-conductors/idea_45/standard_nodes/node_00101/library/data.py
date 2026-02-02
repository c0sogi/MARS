import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.features import get_data_loaders


class SelectiveScaler:
    """
    Scales specific columns of the input data using StandardScaler,
    leaving other columns (e.g., One-Hot encodings) unchanged.
    """

    def __init__(self, scale_indices):
        self.scale_indices = scale_indices
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        if not self.scale_indices:
            return
        self.scaler.fit(X[:, self.scale_indices])
        self.is_fitted = True

    def transform(self, X):
        if not self.scale_indices or not self.is_fitted:
            return X
        X_new = X.copy()
        X_new[:, self.scale_indices] = self.scaler.transform(X[:, self.scale_indices])
        return X_new

    def save(self, path):
        if not self.is_fitted:
            return
        np.savez(
            path, mean=self.scaler.mean_, scale=self.scaler.scale_, var=self.scaler.var_
        )

    def load(self, path):
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.scaler.mean_ = data["mean"]
        self.scaler.scale_ = data["scale"]
        self.scaler.var_ = data["var"]
        self.is_fitted = True
        return True


class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for material crystals.
    """

    def __init__(self, data_dict):
        self.X_atomic = torch.from_numpy(data_dict["X_atomic"]).float()
        self.X_global = torch.from_numpy(data_dict["X_global"]).float()
        self.y = torch.from_numpy(data_dict["y"]).float()

        # Pre-calculate slices for each sample based on batch_idx
        # batch_idx maps atom_index -> sample_index
        batch_idx = data_dict["batch_idx"]

        # Count atoms per sample
        # Assuming batch_idx is sorted and contiguous 0..N-1
        counts = np.bincount(batch_idx)

        # Calculate start indices
        cumulative = np.cumsum(counts)
        starts = np.concatenate(([0], cumulative[:-1]))

        self.slices = [slice(s, s + c) for s, c in zip(starts, counts)]
        self.length = len(counts)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Slice atomic features for this crystal
        sl = self.slices[idx]
        x_atomic = self.X_atomic[sl]
        x_global = self.X_global[idx]
        target = self.y[idx]
        return x_atomic, x_global, target


def collate_sparse(batch):
    """
    Custom collate function for sparse batching.
    Stacks atomic features into a single tensor and creates a batch_idx mapping.
    """
    x_atomic_list, x_global_list, y_list = zip(*batch)

    # Global features and targets are just stacked
    batch_x_global = torch.stack(x_global_list)
    batch_y = torch.stack(y_list)

    # Atomic features are concatenated
    batch_x_atomic = torch.cat(x_atomic_list, dim=0)

    # Create batch_idx vector
    # Maps each atom in the batch to the sample index (0 to batch_size-1)
    batch_idx_list = []
    for i, x_a in enumerate(x_atomic_list):
        n_atoms = x_a.shape[0]
        batch_idx_list.append(torch.full((n_atoms,), i, dtype=torch.long))

    batch_idx = torch.cat(batch_idx_list, dim=0)

    return batch_x_atomic, batch_x_global, batch_y, batch_idx


def get_loaders(
    batch_size=Config.BATCH_SIZE, debug_size=None, load_cached_scalers=True
):
    """
    Orchestrates data loading, scaling, and DataLoader creation.

    Args:
        batch_size (int): Batch size for training/inference.
        debug_size (int): Optional number of samples to load for debugging.
        load_cached_scalers (bool): Whether to try loading scalers from disk.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load raw processed data (uses features.py caching)
    train_data, val_data, test_data = get_data_loaders(debug_size=debug_size)

    # 2. Log Transform Targets (log1p)
    # We apply this in-place to the numpy arrays before creating Datasets
    train_data["y"] = np.log1p(train_data["y"])
    val_data["y"] = np.log1p(val_data["y"])
    # Test targets are placeholders, but transform for consistency/safety
    test_data["y"] = np.log1p(test_data["y"])

    # 3. Initialize Scalers
    atomic_scaler = SelectiveScaler(Config.ATOMIC_SCALE_INDICES)
    global_scaler = SelectiveScaler(Config.GLOBAL_SCALE_INDICES)

    # 4. Fit or Load Scalers
    scaler_loaded = False
    if load_cached_scalers and os.path.exists(Config.SCALER_PATH):
        try:
            # We save both scalers in one npz file for simplicity
            data = np.load(Config.SCALER_PATH)

            # Manually load state into objects
            atomic_scaler.scaler.mean_ = data["atomic_mean"]
            atomic_scaler.scaler.scale_ = data["atomic_scale"]
            atomic_scaler.scaler.var_ = data["atomic_var"]
            atomic_scaler.is_fitted = True

            global_scaler.scaler.mean_ = data["global_mean"]
            global_scaler.scaler.scale_ = data["global_scale"]
            global_scaler.scaler.var_ = data["global_var"]
            global_scaler.is_fitted = True

            scaler_loaded = True
            print("Loaded scalers from cache.")
        except Exception as e:
            print(f"Failed to load scaler cache: {e}")

    if not scaler_loaded:
        print("Fitting scalers on training data...")
        atomic_scaler.fit(train_data["X_atomic"])
        global_scaler.fit(train_data["X_global"])

        # Save scalers
        np.savez(
            Config.SCALER_PATH,
            atomic_mean=atomic_scaler.scaler.mean_,
            atomic_scale=atomic_scaler.scaler.scale_,
            atomic_var=atomic_scaler.scaler.var_,
            global_mean=global_scaler.scaler.mean_,
            global_scale=global_scaler.scaler.scale_,
            global_var=global_scaler.scaler.var_,
        )
        print(f"Saved scalers to {Config.SCALER_PATH}")

    # 5. Apply Scaling
    train_data["X_atomic"] = atomic_scaler.transform(train_data["X_atomic"])
    train_data["X_global"] = global_scaler.transform(train_data["X_global"])

    val_data["X_atomic"] = atomic_scaler.transform(val_data["X_atomic"])
    val_data["X_global"] = global_scaler.transform(val_data["X_global"])

    test_data["X_atomic"] = atomic_scaler.transform(test_data["X_atomic"])
    test_data["X_global"] = global_scaler.transform(test_data["X_global"])

    # 6. Create Datasets
    train_dataset = MaterialsDataset(train_data)
    val_dataset = MaterialsDataset(val_data)
    test_dataset = MaterialsDataset(test_data)

    # 7. Create DataLoaders
    # Use shuffle=True for train, False for val/test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sparse,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sparse,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sparse,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
