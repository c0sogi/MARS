import random
import os
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StandardScaler:
    """
    StandardScaler for normalizing features to zero mean and unit variance.
    Supports saving and loading state for inference.
    """

    def __init__(self, mean=None, std=None, epsilon=1e-8):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, data):
        """
        Computes mean and std from the data.
        Args:
            data: Numpy array of shape (N, D)
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero for constant features
        self.std[self.std < self.epsilon] = 1.0
        return self

    def transform(self, data):
        """
        Standardizes the data using pre-computed mean and std.
        Args:
            data: Numpy array of shape (N, D)
        Returns:
            Scaled data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, data):
        """
        Fits the scaler and transforms the data.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Scales the data back to original distribution.
        Args:
            data: Scaled numpy array
        Returns:
            Original scale data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data * self.std) + self.mean

    def save(self, path):
        """
        Saves the scaler parameters to a .npz file.
        """
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        """
        Loads the scaler parameters from a .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        return self


def log_transform(y):
    """
    Applies log(1 + y) transformation to targets.
    Useful for regression targets that are positive and have a long tail,
    and aligns MSE loss with RMSLE metric.
    """
    return np.log1p(y)


def inverse_log_transform(y_pred):
    """
    Applies exp(y) - 1 transformation to predictions.
    Reverses the log_transform.
    """
    return np.expm1(y_pred)
