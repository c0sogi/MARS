import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for the Seismic Eruption Prediction task.

    This dataset handles the loading of pre-computed spectrograms and targets.
    It implements the 'Latent-Source Cepstral Stacking' strategy requirement
    of Log-Scaling the target variable to ensure neural network convergence.
    """

    def __init__(self, spectrograms, targets, mode="train"):
        """
        Args:
            spectrograms (np.ndarray): Input data of shape (N, 10, 128, T).
            targets (np.ndarray): Target values of shape (N,).
            mode (str): 'train', 'val', or 'test'.
        """
        self.spectrograms = spectrograms
        self.targets = targets
        self.mode = mode
        self.config = Config()

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.spectrograms)

    def __getitem__(self, idx):
        """
        Retrieves the spectrogram and target for a given index.

        Applies np.log1p scaling to the target if configured, which is critical
        for the Vision Branch to handle the large dynamic range of eruption times.
        """
        # Load spectrogram: Shape (10, 128, T)
        # Ensure it is a float32 tensor
        spec = torch.tensor(self.spectrograms[idx], dtype=torch.float32)

        # Load target
        target_val = self.targets[idx]

        # Apply Log-Scaling if configured
        # This compresses the target range (0 to ~4e7) to a learnable range (0 to ~17.5)
        if self.config.NN_PARAMS["target_log_scale"]:
            # Ensure non-negative input for log1p (though time is always >= 0)
            target_val = np.log1p(max(0.0, target_val))

        target = torch.tensor(target_val, dtype=torch.float32)

        return spec, target


def get_data_loaders(train_specs, train_targets, val_specs, val_targets):
    """
    Creates and returns PyTorch DataLoaders for training and validation.

    Args:
        train_specs (np.ndarray): Training spectrograms.
        train_targets (np.ndarray): Training targets.
        val_specs (np.ndarray): Validation spectrograms.
        val_targets (np.ndarray): Validation targets.

    Returns:
        tuple: (train_loader, val_loader)
    """
    config = Config()

    # Instantiate Datasets
    train_dataset = VolcanoDataset(train_specs, train_targets, mode="train")
    val_dataset = VolcanoDataset(val_specs, val_targets, mode="val")

    # Determine device-specific settings
    pin_memory = True if config.NN_PARAMS["device"] == "cuda" else False

    # Create Training DataLoader
    # Shuffle is True to ensure random sampling
    # drop_last is True to avoid unstable gradients from small final batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.NN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=config.NN_PARAMS["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Create Validation DataLoader
    # Shuffle is False for consistent evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.NN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=config.NN_PARAMS["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
