import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FoldScaler:
    """
    Implements independent per-channel Min-Max scaling.
    Designed to fit on training data and transform validation/test data
    using the training statistics to prevent data leakage.
    """

    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.is_fitted = False

    def fit(self, X):
        """
        Computes the min and max values for each channel in the input dataset.

        Args:
            X (np.ndarray): Input data of shape (N, C, H, W).

        Returns:
            self: Returns the instance itself.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim != 4:
            raise ValueError(
                f"Expected input with 4 dimensions (N, C, H, W), got {X.shape}"
            )

        # Calculate min and max across Batch (0), Height (2), and Width (3) dimensions.
        # This preserves the Channel (1) dimension.
        # Resulting shape: (1, C, 1, 1)
        self.min_vals = np.min(X, axis=(0, 2, 3), keepdims=True)
        self.max_vals = np.max(X, axis=(0, 2, 3), keepdims=True)

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Scales the input data using the fitted min and max values.
        Formula: (X - min) / (max - min)

        Args:
            X (np.ndarray): Input data of shape (N, C, H, W).

        Returns:
            np.ndarray: Scaled data with values between 0 and 1.
        """
        if not self.is_fitted:
            raise RuntimeError("FoldScaler must be fitted before calling transform.")

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim != 4:
            raise ValueError(
                f"Expected input with 4 dimensions (N, C, H, W), got {X.shape}"
            )

        # Calculate denominator (range), handling potential division by zero
        denom = self.max_vals - self.min_vals
        denom = np.where(denom == 0, 1.0, denom)

        # Apply scaling
        X_scaled = (X - self.min_vals) / denom

        return X_scaled.astype(np.float32)

    def fit_transform(self, X):
        """
        Fits the scaler to X and returns the transformed X.

        Args:
            X (np.ndarray): Input data of shape (N, C, H, W).

        Returns:
            np.ndarray: Scaled data.
        """
        return self.fit(X).transform(X)
