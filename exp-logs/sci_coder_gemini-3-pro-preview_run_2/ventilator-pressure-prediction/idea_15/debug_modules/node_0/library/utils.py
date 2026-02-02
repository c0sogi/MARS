import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that assigns different weights to the inspiratory
    and expiratory phases of the breath as defined in the configuration.
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.w_insp = Config.LOSS_WEIGHT_INSPIRATORY
        self.w_exp = Config.LOSS_WEIGHT_EXPIRATORY

    def forward(self, preds, targets, u_out):
        """
        Calculates the weighted L1 loss.

        Args:
            preds (torch.Tensor): Predicted pressures.
            targets (torch.Tensor): Actual pressures.
            u_out (torch.Tensor): Control input indicating phase (0: insp, 1: exp).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure u_out matches preds dtype for calculation
        if u_out.dtype != preds.dtype:
            u_out = u_out.to(dtype=preds.dtype)

        # Calculate element-wise absolute error
        abs_error = torch.abs(preds - targets)

        # Determine weights based on u_out
        # u_out is 0 for inspiratory, 1 for expiratory
        # Weight = w_insp if u_out=0, w_exp if u_out=1
        weights = (1 - u_out) * self.w_insp + u_out * self.w_exp

        # Apply weights
        weighted_error = abs_error * weights

        # Return mean loss
        return weighted_error.mean()


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) for the inspiratory phase only.
    This is the official competition metric.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted pressures.
        targets (torch.Tensor or np.ndarray): Actual pressures.
        u_out (torch.Tensor or np.ndarray): Control input (0: insp, 1: exp).

    Returns:
        float: MAE for the inspiratory phase.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure all tensors are on the same device
    if preds.device != u_out.device:
        u_out = u_out.to(preds.device)
    if targets.device != preds.device:
        targets = targets.to(preds.device)

    # Create mask for inspiratory phase (u_out == 0)
    # u_out is typically 0 or 1.
    mask = u_out == 0

    # Select only inspiratory phase elements
    preds_insp = torch.masked_select(preds, mask)
    targets_insp = torch.masked_select(targets, mask)

    # Handle case with no inspiratory steps (unlikely in valid batches but possible)
    if preds_insp.numel() == 0:
        return 0.0

    # Calculate MAE
    mae = torch.abs(preds_insp - targets_insp).mean()

    return mae.item()


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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
