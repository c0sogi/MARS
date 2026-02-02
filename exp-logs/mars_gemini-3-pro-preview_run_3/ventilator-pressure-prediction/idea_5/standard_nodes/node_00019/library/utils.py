import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures CuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metric(y_pred, y_true, u_out):
    """
    Calculates the Mean Absolute Error (MAE) for the inspiratory phase.
    The inspiratory phase is defined as time steps where u_out == 0.
    The expiratory phase (u_out == 1) is ignored.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted pressures.
        y_true (torch.Tensor or np.ndarray): Ground truth pressures.
        u_out (torch.Tensor or np.ndarray): Control input u_out (0 for inspiratory, 1 for expiratory).

    Returns:
        float: The MAE score for the inspiratory phase.
    """
    # Convert inputs to tensors if they are numpy arrays
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Move to CPU for metric calculation to avoid unnecessary GPU usage for scalar logic
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()
    u_out = u_out.detach().cpu()

    # Create boolean mask for inspiratory phase (u_out == 0)
    # Using a threshold or direct comparison depending on data type,
    # but direct comparison is safe given u_out is binary 0/1.
    mask = u_out == 0

    # Check if mask is empty (edge case handling)
    if mask.sum() == 0:
        return 0.0

    # Apply mask to select only inspiratory phase data
    y_pred_masked = y_pred[mask]
    y_true_masked = y_true[mask]

    # Compute Mean Absolute Error
    mae = torch.abs(y_pred_masked - y_true_masked).mean()

    return mae.item()
