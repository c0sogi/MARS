import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.utils import get_data


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type dataset.
    Wraps numpy arrays for features, targets, and IDs.
    """

    def __init__(self, X, y=None, ids=None):
        # Use torch.from_numpy to avoid copying data where possible
        # Data is already cast to correct types in library.utils
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None
        self.ids = torch.from_numpy(ids) if ids is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        elif self.ids is not None:
            return self.X[idx], self.ids[idx]
        else:
            return self.X[idx]


def get_dataloaders(
    batch_size=4096,
    num_workers=4,
    load_cached_data=True,
    cache_dir="./working/idea_39/",
    metadata_dir="./metadata",
):
    """
    Loads data using library.utils.get_data and returns PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size for training and inference.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to load from cache.
        cache_dir (str): Directory for caching processed numpy files.
        metadata_dir (str): Directory containing metadata parquets.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim, num_classes)
    """
    # Load data (cached or processed)
    data = get_data(
        load_cached_data=load_cached_data,
        cache_dir=cache_dir,
        metadata_dir=metadata_dir,
    )

    # Create Datasets
    train_ds = CoverTypeDataset(data["train_X"], data["train_y"])
    val_ds = CoverTypeDataset(data["val_X"], data["val_y"])
    test_ds = CoverTypeDataset(data["test_X"], ids=data["test_ids"])

    # Create DataLoaders
    # Pin memory speeds up host-to-device transfer on GPU
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Determine dimensions
    input_dim = data["train_X"].shape[1]

    # Calculate number of classes based on max label index + 1.
    # Labels are 0-indexed in get_data (original 1-7 mapped to 0-6).
    # We ensure the output layer covers the full range of indices.
    num_classes = int(np.max(data["train_y"]) + 1)

    return train_loader, val_loader, test_loader, input_dim, num_classes
