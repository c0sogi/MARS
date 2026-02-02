import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    # Enable benchmark for performance as input size is fixed (80 steps)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    Automatically selects the available GPU or falls back to CPU.

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_metric(
    y_pred: torch.Tensor, y_true: torch.Tensor, u_out: torch.Tensor
) -> float:
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The competition metric only scores time steps where u_out == 0.

    Args:
        y_pred (torch.Tensor): Predicted pressure values.
        y_true (torch.Tensor): Actual pressure values.
        u_out (torch.Tensor): Control input indicating expiratory phase (1) or inspiratory phase (0).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Ensure inputs are on the same device and flattened if necessary
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    u_out = u_out.view(-1)

    # Create mask for inspiratory phase (u_out == 0)
    # u_out is binary (0 or 1), so we can treat it as a boolean mask
    mask = u_out == 0

    # Filter predictions and targets
    y_pred_insp = y_pred[mask]
    y_true_insp = y_true[mask]

    if len(y_true_insp) == 0:
        return 0.0

    # Calculate MAE
    loss = torch.abs(y_pred_insp - y_true_insp).mean()

    return loss.item()
