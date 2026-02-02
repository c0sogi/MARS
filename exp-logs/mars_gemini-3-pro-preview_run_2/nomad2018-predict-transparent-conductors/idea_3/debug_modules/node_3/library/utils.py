import logging
import sys
import numpy as np
import torch


def setup_logger(name="db_gt", log_file=None, level=logging.INFO):
    """
    Sets up a logger with the specified name, file, and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def compute_rmsle(y_true, y_pred):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true: Ground truth values (numpy array or torch tensor).
        y_pred: Predicted values (numpy array or torch tensor).

    Returns:
        float: The mean RMSLE across all target columns.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to be non-negative to avoid log errors
    # Targets (formation energy and bandgap) are observed to be >= 0 in data analysis
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Compute squared log error
    # log1p(x) = log(1 + x)
    squared_log_error = (np.log1p(y_pred) - np.log1p(y_true)) ** 2

    # Mean over samples (axis 0) to get MSE per column
    mean_squared_log_error = np.mean(squared_log_error, axis=0)

    # Root of MSE per column
    rmsle_per_column = np.sqrt(mean_squared_log_error)

    # Mean across columns
    return np.mean(rmsle_per_column)


class TargetScaler:
    """
    Standardizes target variables by removing the mean and scaling to unit variance.
    Supports inverse transformation to get back original units.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, y):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            y (np.ndarray or torch.Tensor): The data used to compute the mean and standard deviation.
                                            Shape: (n_samples, n_features)
        """
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        self.mean = np.mean(y, axis=0)
        self.std = np.std(y, axis=0)

        # Avoid division by zero
        self.std[self.std == 0] = 1.0

        self._fitted = True

    def transform(self, y):
        """
        Perform standardization by centering and scaling.

        Args:
            y (np.ndarray or torch.Tensor): The data to transform.

        Returns:
            The transformed data (same type as input).
        """
        if not self._fitted:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        is_tensor = isinstance(y, torch.Tensor)
        if is_tensor:
            device = y.device
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y

        y_scaled = (y_np - self.mean) / self.std

        if is_tensor:
            return torch.tensor(y_scaled, dtype=torch.float32, device=device)
        return y_scaled

    def inverse_transform(self, y):
        """
        Scale back the data to the original representation.

        Args:
            y (np.ndarray or torch.Tensor): The data to inverse transform.

        Returns:
            The inverse transformed data (same type as input).
        """
        if not self._fitted:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        is_tensor = isinstance(y, torch.Tensor)
        if is_tensor:
            device = y.device
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y

        y_orig = (y_np * self.std) + self.mean

        if is_tensor:
            return torch.tensor(y_orig, dtype=torch.float32, device=device)
        return y_orig

    def state_dict(self):
        """Returns the state of the scaler."""
        return {"mean": self.mean, "std": self.std, "fitted": self._fitted}

    def load_state_dict(self, state):
        """Loads the state of the scaler."""
        self.mean = state["mean"]
        self.std = state["std"]
        self._fitted = state["fitted"]
