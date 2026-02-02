import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StandardScaler:
    """
    Standard Scaler for Z-score normalization of data.
    Supports both numpy arrays and PyTorch tensors.
    Can save and load its state (mean and std) to/from .npz files.
    """

    def __init__(self, mean=None, std=None, epsilon=1e-8):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, data):
        """
        Compute mean and standard deviation from the data.

        Args:
            data (np.ndarray): Input data of shape (N, D) or (N,).
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero by replacing small std values with 1.0
        self.std = np.where(self.std < self.epsilon, 1.0, self.std)

    def transform(self, data):
        """
        Apply Z-score normalization: (data - mean) / std.

        Args:
            data (np.ndarray or torch.Tensor): Input data.

        Returns:
            Scaled data (same type as input).
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if isinstance(data, torch.Tensor):
            device = data.device
            mean_t = torch.tensor(self.mean, dtype=data.dtype, device=device)
            std_t = torch.tensor(self.std, dtype=data.dtype, device=device)
            return (data - mean_t) / (std_t + self.epsilon)
        else:
            return (data - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, data):
        """
        Reverse Z-score normalization: (data * std) + mean.

        Args:
            data (np.ndarray or torch.Tensor): Scaled data.

        Returns:
            Original scale data (same type as input).
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if isinstance(data, torch.Tensor):
            device = data.device
            mean_t = torch.tensor(self.mean, dtype=data.dtype, device=device)
            std_t = torch.tensor(self.std, dtype=data.dtype, device=device)
            return (data * std_t) + mean_t
        else:
            return (data * self.std) + self.mean

    def save(self, path):
        """
        Save scaler state (mean and std) to a .npz file.

        Args:
            path (str): Path to save the .npz file.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted, cannot save.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        """
        Load scaler state from a .npz file.

        Args:
            path (str): Path to the .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


def get_scaler(data, path, load_cached_data=True):
    """
    Helper function to get a fitted scaler.
    It tries to load from cache first; if failed or not requested, it fits on data and saves.

    Args:
        data (np.ndarray): Data to fit on if cache is not used. Can be None if loading from cache.
        path (str): Path to the cached .npz file.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        StandardScaler: A fitted scaler object.
    """
    scaler = StandardScaler()

    # Try to load from cache
    if load_cached_data and os.path.exists(path):
        try:
            scaler.load(path)
            return scaler
        except Exception:
            # If load fails, fall through to fit
            pass

    # If we are here, we need to fit the scaler
    if data is None:
        raise ValueError(
            f"Cache not found at {path} and no data provided to fit scaler."
        )

    scaler.fit(data)
    scaler.save(path)
    return scaler


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch.
        loss (float): Current validation loss.
        path (str): Path to save the .pth file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into (can be None).
        path (str): Path to the .pth file.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        tuple: (epoch, loss) from the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss
