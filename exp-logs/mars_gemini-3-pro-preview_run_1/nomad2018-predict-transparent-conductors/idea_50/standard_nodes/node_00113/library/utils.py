import os
import random
import numpy as np
import torch


def set_seed(seed):
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
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log1p_transform(y):
    """
    Applies log(1 + y) transformation to targets.
    Useful for regression targets that are positive and skewed.

    Args:
        y (np.ndarray): Target values.

    Returns:
        np.ndarray: Log-transformed values.
    """
    return np.log1p(y)


def expm1_transform(y):
    """
    Applies exp(y) - 1 transformation to predictions.
    Inverse of log1p_transform.

    Args:
        y (np.ndarray): Log-transformed values.

    Returns:
        np.ndarray: Original scale values.
    """
    return np.expm1(y)


class SelectiveScaler:
    """
    Standard Scaler that scales only specific columns (continuous features)
    while leaving others (e.g., one-hot encoded features) untouched.
    """

    def __init__(self, cols_to_scale=None):
        """
        Args:
            cols_to_scale (list of int, optional): Indices of columns to scale.
                                                   If None, all columns are scaled.
        """
        self.cols_to_scale = cols_to_scale
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            X (np.ndarray): Data to fit, shape (n_samples, n_features).

        Returns:
            self: Returns the instance itself.
        """
        if self.cols_to_scale is None:
            self.cols_to_scale = list(range(X.shape[1]))

        # Extract data to scale
        X_cont = X[:, self.cols_to_scale]

        self.mean_ = np.mean(X_cont, axis=0)
        self.scale_ = np.std(X_cont, axis=0)

        # Prevent division by zero for constant features
        self.scale_[self.scale_ == 0] = 1.0

        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling on selected columns.

        Args:
            X (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        X_transformed = X.copy()

        # Scale selected columns
        X_cont = X_transformed[:, self.cols_to_scale]
        X_cont = (X_cont - self.mean_) / self.scale_
        X_transformed[:, self.cols_to_scale] = X_cont

        return X_transformed

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Data to fit and transform.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.fit(X).transform(X)

    def save(self, path):
        """
        Save scaler statistics to a .npz file.

        Args:
            path (str): File path to save the scaler state.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted, cannot save.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(
            path,
            mean=self.mean_,
            scale=self.scale_,
            cols_to_scale=np.array(self.cols_to_scale),
        )

    def load(self, path):
        """
        Load scaler statistics from a .npz file.

        Args:
            path (str): File path to load the scaler state from.

        Returns:
            self: Returns the instance with loaded state.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean_ = data["mean"]
        self.scale_ = data["scale"]
        self.cols_to_scale = data["cols_to_scale"].tolist()
        return self
