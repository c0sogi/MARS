import random
import numpy as np
import torch
import os
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    Utility class to standardize target variables (zero mean, unit variance)
    and inverse transform predictions.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Computes mean and standard deviation from the data y.
        y: numpy array or torch tensor of shape (N, num_targets)
        """
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        self.mean = np.mean(y, axis=0)
        self.std = np.std(y, axis=0)

        # Prevent division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, y):
        """
        Standardizes y using the fitted mean and std.
        Returns a torch tensor if input is tensor, else numpy array.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        if isinstance(y, torch.Tensor):
            device = y.device
            y_np = y.cpu().numpy()
            y_scaled = (y_np - self.mean) / self.std
            return torch.tensor(y_scaled, dtype=torch.float32, device=device)
        else:
            return (y - self.mean) / self.std

    def inverse_transform(self, y):
        """
        Inverse transforms standardized data y back to original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        if isinstance(y, torch.Tensor):
            device = y.device
            y_np = y.cpu().numpy()
            y_inv = y_np * self.std + self.mean
            return torch.tensor(y_inv, dtype=torch.float32, device=device)
        else:
            return y * self.std + self.mean

    def save(self, path=Config.TARGET_SCALER_CACHE):
        """
        Saves the scaler parameters (mean and std) to a .npz file.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted, cannot save.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path=Config.TARGET_SCALER_CACHE):
        """
        Loads the scaler parameters from a .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"TargetScaler cache not found at {path}")

        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
