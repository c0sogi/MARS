import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device to be used for training and inference.

    Returns:
        torch.device: The configured device.
    """
    return torch.device(Config.DEVICE)


def compute_metric(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) between predicted and actual pressures,
    considering only the inspiratory phase (where u_out == 0).

    Args:
        y_pred (torch.Tensor or np.ndarray): The predicted pressure values.
        y_true (torch.Tensor or np.ndarray): The ground truth pressure values.
        u_out (torch.Tensor or np.ndarray): The expiratory valve control input (0 or 1).

    Returns:
        float: The MAE score for the inspiratory phase.
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure all tensors are on the same device as predictions
    device = y_pred.device
    if y_true.device != device:
        y_true = y_true.to(device)
    if u_out.device != device:
        u_out = u_out.to(device)

    # Flatten inputs to (N,) to handle both (Batch, Seq) and flattened inputs uniformly
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    u_out = u_out.reshape(-1)

    # Create a boolean mask for the inspiratory phase (u_out == 0)
    # We use a threshold of 0.5 to safely handle potential float representations of binary data
    mask = u_out < 0.5

    # Apply mask to select only inspiratory phase data points
    valid_preds = y_pred[mask]
    valid_true = y_true[mask]

    if valid_true.numel() == 0:
        return 0.0

    # Calculate Mean Absolute Error
    mae = torch.abs(valid_preds - valid_true).mean()

    return mae.item()
