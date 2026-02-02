import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rmsle(y_pred, y_true):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error.

    Args:
        y_pred: Predicted values (Torch Tensor or Numpy Array)
        y_true: Ground truth values (Torch Tensor or Numpy Array)

    Returns:
        float: The mean RMSLE across all target columns.
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Ensure inputs are on CPU for metric calculation
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()

    # Clamp predictions to be non-negative as log is undefined for negative values
    # Formation Energy and Bandgap Energy are physically non-negative in this dataset context
    y_pred = torch.clamp(y_pred, min=0.0)
    y_true = torch.clamp(y_true, min=0.0)

    log_pred = torch.log1p(y_pred)
    log_true = torch.log1p(y_true)

    squared_log_error = (log_pred - log_true) ** 2

    # Mean squared error per column
    mse_per_column = torch.mean(squared_log_error, dim=0)

    # Root mean squared error per column
    rmsle_per_column = torch.sqrt(mse_per_column)

    # Average across columns (Column-wise RMSLE)
    mean_rmsle = torch.mean(rmsle_per_column)

    return mean_rmsle.item()


class StandardScaler:
    """
    StandardScaler that normalizes data to zero mean and unit variance.
    Supports PyTorch tensors and saving/loading state via Numpy to avoid pickle.
    """

    def __init__(self, device=None):
        self.mean = None
        self.std = None
        self.device = device if device else torch.device("cpu")

    def fit(self, data):
        """
        Computes mean and standard deviation from the data.

        Args:
            data (torch.Tensor or np.ndarray): The input data to fit.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        data = data.to(self.device)
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

        # Handle constant features (std=0) to avoid division by zero
        # Replace 0 with 1 to keep values unchanged after subtraction of mean
        self.std = torch.where(
            self.std == 0, torch.tensor(1.0, device=self.device), self.std
        )

    def transform(self, data):
        """
        Standardizes the data using pre-computed mean and std.

        Args:
            data (torch.Tensor or np.ndarray): The data to transform.

        Returns:
            torch.Tensor: The standardized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        data = data.to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverses the standardization to get original scale.

        Args:
            data (torch.Tensor or np.ndarray): The standardized data.

        Returns:
            torch.Tensor: The data in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        data = data.to(self.device)
        return (data * self.std) + self.mean

    def save(self, path):
        """
        Saves the mean and std to a .npz file using numpy.

        Args:
            path (str): The file path to save the scaler state.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save unfitted StandardScaler.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Ensure path ends with .npz
        if not path.endswith(".npz"):
            path += ".npz"

        # Save as numpy arrays
        np.savez(
            path,
            mean=self.mean.detach().cpu().numpy(),
            std=self.std.detach().cpu().numpy(),
        )

    def load(self, path):
        """
        Loads the mean and std from a .npz file.

        Args:
            path (str): The file path to load the scaler state from.
        """
        # Ensure path ends with .npz
        if not path.endswith(".npz"):
            path += ".npz"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)
