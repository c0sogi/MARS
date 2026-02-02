import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across standard Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    Manages the scaling (Z-score normalization) of the target variable.
    Implements fit, transform, inverse_transform, and persistence via .npy files.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Computes mean and std from the provided data.

        Args:
            y (array-like): The target values to fit.
        """
        y = np.array(y)
        self.mean = np.mean(y)
        self.std = np.std(y)

        # Prevent division by zero if std is 0 (unlikely for this task but safe)
        if self.std == 0:
            self.std = 1.0

    def transform(self, y):
        """
        Standardizes the input y using the fitted mean and std.
        z = (y - mean) / std

        Args:
            y (array-like or Tensor): The data to transform.

        Returns:
            The transformed data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("TargetScaler has not been fitted yet.")
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        """
        Reverts the standardization.
        x = z * std + mean

        Args:
            y (array-like or Tensor): The data to inverse transform.

        Returns:
            The data in original scale.
        """
        if self.mean is None or self.std is None:
            raise ValueError("TargetScaler has not been fitted yet.")
        return (y * self.std) + self.mean

    def save(self, mean_path=Config.TARGET_MEAN_PATH, std_path=Config.TARGET_STD_PATH):
        """
        Saves the fitted mean and std to .npy files.

        Args:
            mean_path (str): Path to save the mean.
            std_path (str): Path to save the std.
        """
        if self.mean is None or self.std is None:
            raise ValueError("TargetScaler has not been fitted yet.")

        os.makedirs(os.path.dirname(mean_path), exist_ok=True)
        os.makedirs(os.path.dirname(std_path), exist_ok=True)

        np.save(mean_path, self.mean)
        np.save(std_path, self.std)

    def load(self, mean_path=Config.TARGET_MEAN_PATH, std_path=Config.TARGET_STD_PATH):
        """
        Loads mean and std from .npy files.

        Args:
            mean_path (str): Path to load the mean from.
            std_path (str): Path to load the std from.
        """
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(
                f"Scaler files not found at {mean_path} or {std_path}"
            )

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)


def save_checkpoint(model, path):
    """
    Saves the PyTorch model state dictionary.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    """
    Loads the PyTorch model state dictionary.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): The file path to load from.
        device (str or torch.device): The device to map the location to.

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics (Loss, MAE) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
