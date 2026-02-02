import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import engineer_features


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.

    Stores features, control states, targets, and IDs as tensors.
    """

    def __init__(self, x, u_out, y=None, ids=None):
        """
        Args:
            x (np.ndarray): Input features of shape (N, 80, Features).
            u_out (np.ndarray): Control input u_out of shape (N, 80).
            y (np.ndarray, optional): Target pressure of shape (N, 80).
            ids (np.ndarray, optional): Time step IDs of shape (N, 80).
        """
        # Convert to tensors immediately
        self.x = torch.tensor(x, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)

        # Handle targets (y)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            # Create dummy targets for inference if not provided
            self.y = torch.zeros_like(self.u_out, dtype=torch.float32)

        # Handle IDs
        if ids is not None:
            self.ids = torch.tensor(ids, dtype=torch.int32)
        else:
            # Create dummy IDs for training if not provided
            self.ids = torch.zeros_like(self.u_out, dtype=torch.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (x, u_out, y, ids)
        """
        return self.x[idx], self.u_out[idx], self.y[idx], self.ids[idx]


def prepare_datasets(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, force_recompute=False
):
    """
    Loads data, handles caching/debugging logic, and creates PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for loading.
        force_recompute (bool): If True, ignores cache and re-runs feature engineering.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # 1. Load Data via Feature Engineering Pipeline
    # If force_recompute is True, we tell engineer_features NOT to load cached data.
    # This effectively invalidates the cache and overwrites it with new data.
    print(f"Preparing datasets (Force Recompute: {force_recompute})...")
    data = engineer_features(load_cached_data=not force_recompute)

    train_x = data["train_x"]
    train_y = data["train_y"]
    train_u_out = data["train_u_out"]

    val_x = data["val_x"]
    val_y = data["val_y"]
    val_u_out = data["val_u_out"]

    test_x = data["test_x"]
    test_u_out = data["test_u_out"]
    test_ids = data["test_ids"]

    # 2. Handle Debug Mode (Subsampling)
    if Config.DEBUG:
        sample_size = Config.DEBUG_SAMPLE_SIZE
        print(f"DEBUG Mode enabled. Subsampling datasets to {sample_size} breaths.")

        train_x = train_x[:sample_size]
        train_y = train_y[:sample_size]
        train_u_out = train_u_out[:sample_size]

        val_x = val_x[:sample_size]
        val_y = val_y[:sample_size]
        val_u_out = val_u_out[:sample_size]

        test_x = test_x[:sample_size]
        test_u_out = test_u_out[:sample_size]
        test_ids = test_ids[:sample_size]

    # 3. Instantiate Datasets
    train_dataset = VentilatorDataset(train_x, train_u_out, y=train_y)
    val_dataset = VentilatorDataset(val_x, val_u_out, y=val_y)
    test_dataset = VentilatorDataset(test_x, test_u_out, ids=test_ids)

    # 4. Create DataLoaders
    # Pin memory speeds up host-to-device transfer if using GPU
    use_pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,  # Important for BatchNorm stability and consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    print(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    return train_loader, val_loader, test_loader
