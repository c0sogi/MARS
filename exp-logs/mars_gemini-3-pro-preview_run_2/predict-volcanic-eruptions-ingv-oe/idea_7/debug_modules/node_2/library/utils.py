import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    A utility class for scaling target variables using Standard Scaling (Mean/Std).
    Supports saving and loading statistics for consistent inference.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Computes the mean and standard deviation of the target array.

        Args:
            y (np.ndarray or list): The target values.
        """
        y = np.array(y)
        self.mean = np.mean(y)
        self.std = np.std(y)

        # Prevent division by zero if std is 0
        if self.std == 0:
            self.std = 1.0

    def transform(self, y):
        """
        Scales the input data using the computed mean and std.

        Args:
            y (np.ndarray, list, or torch.Tensor): The data to scale.

        Returns:
            The scaled data in the same format as input (numpy or tensor).
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if torch.is_tensor(y):
            return (y - self.mean) / self.std
        else:
            y = np.array(y)
            return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        """
        Reverts the scaling operation.

        Args:
            y_scaled (np.ndarray, list, or torch.Tensor): The scaled data.

        Returns:
            The original scale data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if torch.is_tensor(y_scaled):
            return (y_scaled * self.std) + self.mean
        else:
            y_scaled = np.array(y_scaled)
            return (y_scaled * self.std) + self.mean

    def save(self, mean_path: str, std_path: str):
        """
        Saves the mean and std to numpy files.

        Args:
            mean_path (str): Path to save the mean .npy file.
            std_path (str): Path to save the std .npy file.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        # Ensure directories exist
        os.makedirs(os.path.dirname(mean_path), exist_ok=True)
        os.makedirs(os.path.dirname(std_path), exist_ok=True)

        np.save(mean_path, self.mean)
        np.save(std_path, self.std)

    def load(self, mean_path: str, std_path: str):
        """
        Loads the mean and std from numpy files.

        Args:
            mean_path (str): Path to the mean .npy file.
            std_path (str): Path to the std .npy file.
        """
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(
                f"Scaler files not found at {mean_path} or {std_path}"
            )

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
