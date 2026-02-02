import os
import random
import numpy as np
import torch
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GlobalMinMaxScaler:
    """
    A stateful scaler that normalizes 3-channel image data (HH, HV, Avg)
    to the [0, 1] range using global min/max statistics computed on the training set.
    """

    def __init__(self):
        self.mins = None
        self.maxs = None
        self.fitted = False

    def fit(self, X):
        """
        Computes the min and max values for each channel from the input data X.

        Args:
            X (np.ndarray): Input data. Expected shape is (N, 3, H, W) or (N, H, W, 3).
        """
        X = np.array(X)

        # Determine data layout and compute stats along appropriate axes
        # Case 1: Channel-first (N, C, H, W) -> standard PyTorch layout
        if X.ndim == 4 and X.shape[1] == 3:
            axis = (0, 2, 3)
            # Shape for broadcasting: (3, 1, 1)
            self.mins = X.min(axis=axis).reshape(3, 1, 1)
            self.maxs = X.max(axis=axis).reshape(3, 1, 1)

        # Case 2: Channel-last (N, H, W, C) -> standard Image layout
        elif X.ndim == 4 and X.shape[-1] == 3:
            axis = (0, 1, 2)
            # Shape for broadcasting: (1, 1, 3)
            self.mins = X.min(axis=axis).reshape(1, 1, 3)
            self.maxs = X.max(axis=axis).reshape(1, 1, 3)

        else:
            raise ValueError(
                f"Invalid input shape {X.shape}. Expected (N, 3, H, W) or (N, H, W, 3)."
            )

        self.fitted = True

    def transform(self, X):
        """
        Scales the input X to [0, 1] using the fitted statistics.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled data.
        """
        if not self.fitted:
            raise RuntimeError(
                "GlobalMinMaxScaler must be fitted before calling transform."
            )

        X = np.array(X)

        # Calculate range
        rng = self.maxs - self.mins
        # Prevent division by zero
        rng[rng == 0] = 1.0

        # Apply scaling
        X_scaled = (X - self.mins) / rng

        return X_scaled

    def save(self, save_dir):
        """
        Saves the fitted statistics to a file in the specified directory.
        """
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted GlobalMinMaxScaler.")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "scaler_params.npz")
        np.savez(save_path, mins=self.mins, maxs=self.maxs)

    def load(self, load_dir):
        """
        Loads fitted statistics from the specified directory.
        """
        load_path = os.path.join(load_dir, "scaler_params.npz")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Scaler parameters not found at {load_path}")

        data = np.load(load_path)
        self.mins = data["mins"]
        self.maxs = data["maxs"]
        self.fitted = True


def save_checkpoint(model, path):
    """
    Saves the model's state dictionary to the specified path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device=None):
    """
    Loads the model's state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): Path to the checkpoint file.
        device (torch.device, optional): Device to map the location to.

    Returns:
        model: The model with loaded weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
