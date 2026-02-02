import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import BATCH_SIZE, NUM_WORKERS, SEED
from library.utils import seed_everything


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for handling 10-channel seismic spectrograms.

    Attributes:
        X (np.ndarray): Input spectrograms of shape (N, 10, n_mels, time).
        y (np.ndarray): Target time_to_eruption values of shape (N,).
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve the spectrogram and target
        img = self.X[idx]
        target = self.y[idx]

        # Convert spectrogram to FloatTensor
        # Expected shape by PyTorch CNNs: (Channels, Height, Width)
        # Input is (10, n_mels, time), which aligns correctly.
        img_tensor = torch.from_numpy(img).float()

        # Apply Log-Scaling to the target: y_new = log(1 + y_old)
        # This compresses the large dynamic range of time_to_eruption
        log_target = np.log1p(target)
        target_tensor = torch.tensor(log_target, dtype=torch.float32)

        return img_tensor, target_tensor


def get_data_loaders(
    X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
):
    """
    Creates and returns DataLoaders for training and validation sets.

    Args:
        X_train, y_train: Training data and targets.
        X_val, y_val: Validation data and targets.
        batch_size (int): Batch size for training/validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # Instantiate Datasets
    train_dataset = VolcanoDataset(X_train, y_train)
    val_dataset = VolcanoDataset(X_val, y_val)

    # Create Training DataLoader
    # drop_last=True is used to prevent issues with Batch Normalization on small final batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(X_test, y_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Creates and returns a DataLoader for the test set.

    Args:
        X_test, y_test: Test data and placeholder targets.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of subprocesses.

    Returns:
        DataLoader: The test data loader.
    """
    test_dataset = VolcanoDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
