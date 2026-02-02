import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The inspiratory phase is defined where u_out == 0.

    This function supports both PyTorch tensors and NumPy arrays.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted pressures.
        targets (torch.Tensor or np.ndarray): Ground truth pressures.
        u_out (torch.Tensor or np.ndarray): Control input indicating phase (0=in, 1=out).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert NumPy arrays to PyTorch tensors if necessary
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure all tensors are on the same device as predictions
    device = preds.device
    if targets.device != device:
        targets = targets.to(device)
    if u_out.device != device:
        u_out = u_out.to(device)

    # Flatten tensors to 1D to ensure shape alignment
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    u_out = u_out.reshape(-1)

    # Create mask for the inspiratory phase (u_out == 0)
    # The competition metric only scores the inspiratory phase.
    mask = u_out == 0

    # Apply mask to select only inspiratory time steps
    preds_masked = preds[mask]
    targets_masked = targets[mask]

    # Handle edge case where mask is empty (though unlikely in this dataset)
    if preds_masked.numel() == 0:
        return 0.0

    # Calculate Mean Absolute Error
    loss = torch.abs(preds_masked - targets_masked).mean()

    return loss.item()
