import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN to guarantee fully reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (cuda or cpu) based on availability.

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_metric(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically on the inspiratory phase.
    The inspiratory phase is defined as time steps where u_out == 0.

    Args:
        y_pred: Predicted pressures (torch.Tensor or np.ndarray).
        y_true: Actual pressures (torch.Tensor or np.ndarray).
        u_out: Control input for expiratory valve (torch.Tensor or np.ndarray).
               0 represents inspiratory phase, 1 represents expiratory phase.

    Returns:
        float: The MAE calculated only where u_out == 0.
    """
    # Handle Numpy arrays
    if isinstance(y_pred, np.ndarray):
        # Create mask for inspiratory phase (u_out == 0)
        mask = u_out == 0

        # Filter predictions and targets using the mask
        # Flattening is implicit in boolean indexing for numpy
        y_pred_insp = y_pred[mask]
        y_true_insp = y_true[mask]

        # Avoid division by zero if no inspiratory phase exists (edge case)
        if len(y_pred_insp) == 0:
            return 0.0

        mae = np.mean(np.abs(y_pred_insp - y_true_insp))
        return float(mae)

    # Handle Torch tensors
    elif isinstance(y_pred, torch.Tensor):
        # Ensure inputs are on the same device
        if y_true.device != y_pred.device:
            y_true = y_true.to(y_pred.device)
        if u_out.device != y_pred.device:
            u_out = u_out.to(y_pred.device)

        # Create mask for inspiratory phase (u_out == 0)
        mask = u_out == 0

        # Filter predictions and targets
        y_pred_insp = y_pred[mask]
        y_true_insp = y_true[mask]

        if y_pred_insp.numel() == 0:
            return 0.0

        mae = torch.abs(y_pred_insp - y_true_insp).mean()
        return mae.item()

    else:
        raise TypeError("Inputs must be either numpy arrays or torch tensors.")
