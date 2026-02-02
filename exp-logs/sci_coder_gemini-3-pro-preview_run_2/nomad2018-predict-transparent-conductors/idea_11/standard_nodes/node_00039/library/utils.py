import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
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


def compute_rmsle(y_true, y_pred):
    """
    Computes the Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The RMSLE score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip predictions to be non-negative to avoid log domain errors
    # The target values are non-negative (energy), so predictions should be treated as such.
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Compute RMSLE: sqrt(mean((log(p+1) - log(a+1))^2))
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    mean_squared_log_error = np.mean(np.square(log_diff))
    rmsle = np.sqrt(mean_squared_log_error)

    return rmsle


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the specified filename.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint from the specified filename.

    Args:
        filename (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load the state dict into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str or torch.device): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(filename):
        return None

    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if just the state dict was saved
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Supports saving and loading state for inference.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            data (np.ndarray): The data used to compute the mean and standard deviation.
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Handle zero standard deviation to avoid division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """
        Perform standardization by centering and scaling.

        Args:
            data (np.ndarray): The data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Scale back the data to the original representation.

        Args:
            data (np.ndarray): The data to inverse transform.

        Returns:
            np.ndarray: The original data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data * self.std) + self.mean

    def state_dict(self):
        """Returns the state of the scaler."""
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        """Loads the state of the scaler."""
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
