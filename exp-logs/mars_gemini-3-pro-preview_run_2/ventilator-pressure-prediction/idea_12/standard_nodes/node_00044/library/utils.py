import os
import random
import numpy as np
import torch
import torch.nn as nn


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

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightedL1Loss(nn.Module):
    """
    Custom L1 Loss that assigns different weights to the inspiratory and expiratory phases.

    Strategy:
    - Inspiratory phase (u_out = 0): Weight = 1.0
    - Expiratory phase (u_out = 1): Weight = 0.1
    """

    def __init__(self, inspiratory_weight=1.0, expiratory_weight=0.1):
        super(WeightedL1Loss, self).__init__()
        self.inspiratory_weight = inspiratory_weight
        self.expiratory_weight = expiratory_weight

    def forward(self, preds, targets, u_out):
        """
        Calculates the weighted L1 loss.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth pressure values.
            u_out (torch.Tensor): Control input indicating phase (0 for inspiratory, 1 for expiratory).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Calculate element-wise absolute error
        abs_error = torch.abs(preds - targets)

        # Determine weights based on u_out
        # u_out is 0 or 1.
        # If u_out == 0 (Inspiratory) -> weight = inspiratory_weight
        # If u_out == 1 (Expiratory) -> weight = expiratory_weight
        weights = (1 - u_out) * self.inspiratory_weight + u_out * self.expiratory_weight

        # Apply weights
        weighted_error = abs_error * weights

        # Return mean loss
        return weighted_error.mean()


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) for the inspiratory phase only.

    The competition metric is defined as the MAE between predicted and actual pressures
    during the inspiratory phase of each breath. The expiratory phase is not scored.

    Args:
        preds (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth pressure values.
        u_out (torch.Tensor): Control input indicating phase (0 for inspiratory, 1 for expiratory).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Ensure inputs are on the same device and detached from graph if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach()

    # Create mask for inspiratory phase (u_out == 0)
    # Using boolean indexing
    mask = u_out == 0

    # Filter predictions and targets
    preds_inspiratory = preds[mask]
    targets_inspiratory = targets[mask]

    # Calculate MAE
    if len(targets_inspiratory) == 0:
        return 0.0

    mae = torch.abs(preds_inspiratory - targets_inspiratory).mean()

    return mae.item()
