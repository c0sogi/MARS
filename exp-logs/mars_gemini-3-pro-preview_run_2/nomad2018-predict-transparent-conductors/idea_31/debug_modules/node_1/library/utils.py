import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class StandardScaler:
    """
    StandardScaler for normalizing and denormalizing target variables.
    Implements persistence using numpy to avoid pickle for data attributes.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.device = Config.DEVICE

    def fit(self, data):
        """
        Computes mean and std from the data.

        Args:
            data (torch.Tensor or np.ndarray): The data to fit on.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

        # Handle zero variance to avoid division by zero
        self.std[self.std == 0] = 1.0

        print(f"Scaler fitted. Mean: {self.mean}, Std: {self.std}")

    def transform(self, data):
        """
        Normalizes the data: (x - mean) / std.

        Args:
            data (torch.Tensor or np.ndarray): Data to normalize.

        Returns:
            Normalized data of the same type as input.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if isinstance(data, torch.Tensor):
            device = data.device
            mean_t = torch.tensor(self.mean, device=device, dtype=data.dtype)
            std_t = torch.tensor(self.std, device=device, dtype=data.dtype)
            return (data - mean_t) / std_t
        else:
            return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Denormalizes the data: x * std + mean.

        Args:
            data (torch.Tensor or np.ndarray): Data to denormalize.

        Returns:
            Denormalized data of the same type as input.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if isinstance(data, torch.Tensor):
            device = data.device
            mean_t = torch.tensor(self.mean, device=device, dtype=data.dtype)
            std_t = torch.tensor(self.std, device=device, dtype=data.dtype)
            return data * std_t + mean_t
        else:
            return data * self.std + self.mean

    def save(self, path):
        """
        Saves the scaler state (mean and std) to a .npz file.

        Args:
            path (str): Path to save the .npz file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)
        print(f"Scaler state saved to {path}")

    def load(self, path):
        """
        Loads the scaler state from a .npz file.

        Args:
            path (str): Path to the .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        print(f"Scaler state loaded from {path}")


def compute_metrics(preds, targets):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        preds (torch.Tensor or np.ndarray): Predicted values.
        targets (torch.Tensor or np.ndarray): Ground truth values.

    Returns:
        dict: Dictionary containing 'mean_rmsle' and individual column RMSLEs.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure non-negative values for log1p (physical quantities like energy shouldn't be negative here,
    # but model might output negative values before convergence)
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    # Calculate squared logarithmic errors: (log(p+1) - log(t+1))^2
    log_preds = np.log1p(preds)
    log_targets = np.log1p(targets)
    squared_log_errors = (log_preds - log_targets) ** 2

    # Mean squared log error for each column
    msle_per_column = np.mean(squared_log_errors, axis=0)

    # Root mean squared log error for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Mean RMSLE across columns
    mean_rmsle = np.mean(rmsle_per_column)

    metrics = {"mean_rmsle": mean_rmsle}

    # Add specific keys for the known targets if dimensions match
    if len(rmsle_per_column) >= 1:
        metrics["formation_energy_rmsle"] = rmsle_per_column[0]
    if len(rmsle_per_column) >= 2:
        metrics["bandgap_energy_rmsle"] = rmsle_per_column[1]

    return metrics


def save_checkpoint(model, optimizer, epoch, val_loss, scaler, path):
    """
    Saves the model checkpoint including model state, optimizer state, and scaler state.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch.
        val_loss (float): Validation loss.
        scaler (StandardScaler): The scaler instance.
        path (str): Path to save the checkpoint (.pth).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        # We save scaler state here for convenience in resuming training
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
    }
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, scaler=None):
    """
    Loads the model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into (optional).
        path (str): Path to the checkpoint file.
        scaler (StandardScaler, optional): Scaler to load state into.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return None

    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_mean" in checkpoint:
        scaler.mean = checkpoint["scaler_mean"]
        scaler.std = checkpoint["scaler_std"]
        print("Scaler state loaded from checkpoint.")

    print(f"Checkpoint loaded from {path} (Epoch {checkpoint.get('epoch', -1)})")
    return checkpoint
