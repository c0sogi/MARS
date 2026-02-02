import numpy as np
import os


def transform_targets(y):
    """
    Applies log(1+y) transformation to targets.
    This helps in stabilizing training for targets with large dynamic ranges
    and aligns with the RMSLE metric.
    """
    return np.log1p(y)


def inverse_transform_targets(y):
    """
    Applies exp(y)-1 transformation to targets.
    This is the inverse of transform_targets, used to convert model predictions
    back to the original scale.
    """
    return np.expm1(y)


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            X (np.ndarray): The data used to compute the mean and standard deviation.
        """
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)

        # Handle constant features where std is 0 to avoid division by zero
        # Replace 0 with 1.0 so that division leaves the value (0) unchanged
        self.scale[self.scale < 1e-9] = 1.0
        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Args:
            X (np.ndarray): The data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.

        Args:
            X (np.ndarray): The transformed data.

        Returns:
            np.ndarray: The data in original scale.
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        return X * self.scale + self.mean

    def save(self, path):
        """
        Save the scaler state (mean and scale) to a file.

        Args:
            path (str): The file path to save the .npz file.
        """
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        np.savez(path, mean=self.mean, scale=self.scale)

    def load(self, path):
        """
        Load the scaler state from a file.

        Args:
            path (str): The file path to load the .npz file from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = data["mean"]
        self.scale = data["scale"]
        return self
