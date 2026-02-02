import os
import random
import shutil
import numpy as np
import torch

from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StandardScaler:
    """
    StandardScaler for normalizing and denormalizing target variables.
    Stores mean and standard deviation. Handles data on the specified device.
    """

    def __init__(self, device=Config.DEVICE):
        self.mean = None
        self.std = None
        self.device = device

    def fit(self, y):
        """
        Computes mean and std from the data y.
        y can be a numpy array or torch tensor.
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        self.mean = torch.mean(y, dim=0)
        self.std = torch.std(y, dim=0)

        # Prevent division by zero by replacing 0 std with 1.0
        self.std = torch.where(
            self.std == 0, torch.tensor(1.0, device=self.device), self.std
        )

    def transform(self, y):
        """
        Standardizes y: (y - mean) / std.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler is not fitted yet.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        """
        Reverses standardization: y * std + mean.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler is not fitted yet.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return y * self.std + self.mean

    def save(self, path):
        """
        Saves the mean and std to a .npz file.
        Uses numpy.savez to avoid pickle for data attributes.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler is not fitted, cannot save.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy())

    def load(self, path):
        """
        Loads mean and std from a .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint. If is_best is True, copies it to best_model.pth.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, "best_model.pth"))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.
    Returns the epoch and best_loss if available.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint to the configured device
    # Cite debug_lesson_5: Explicitly Disable weights_only When Loading PyTorch Checkpoints Containing NumPy Arrays
    checkpoint = torch.load(
        checkpoint_path, map_location=Config.DEVICE, weights_only=False
    )

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    return start_epoch, best_loss
