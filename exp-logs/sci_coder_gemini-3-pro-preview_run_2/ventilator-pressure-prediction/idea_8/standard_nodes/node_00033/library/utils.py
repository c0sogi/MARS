import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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
    torch.backends.cudnn.benchmark = False


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss for Ventilator Pressure Prediction.

    Implements a Mean Absolute Error (MAE) loss that assigns different weights
    to the inspiratory and expiratory phases of the breath.

    Strategy:
    - Inspiratory phase (u_out=0): Weight = 1.0 (Competition metric focus)
    - Expiratory phase (u_out=1): Weight = 0.1 (Regularization/Stability focus)
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.w_insp = Config.LOSS_INSPIRATORY_WEIGHT
        self.w_exp = Config.LOSS_EXPIRATORY_WEIGHT

    def forward(self, preds, targets, u_out):
        """
        Calculates the weighted L1 loss.

        Args:
            preds (torch.Tensor): Predicted pressures. Shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1)
            targets (torch.Tensor): Actual pressures. Shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1)
            u_out (torch.Tensor): Expiratory valve control signal (0 or 1).
                                  Shape must match preds/targets broadcast logic.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate element-wise absolute error
        abs_error = torch.abs(preds - targets)

        # Determine weights based on u_out
        # u_out is 0 for inspiratory, 1 for expiratory
        # weight = (1 - u_out) * w_insp + u_out * w_exp
        weights = (1 - u_out) * self.w_insp + u_out * self.w_exp

        # Apply weights
        weighted_error = abs_error * weights

        # Return mean loss
        return weighted_error.mean()


def compute_metric(preds, targets, u_out):
    """
    Calculates the competition metric: Mean Absolute Error (MAE)
    evaluated ONLY during the inspiratory phase (u_out == 0).

    Args:
        preds (torch.Tensor or np.ndarray): Predicted pressures.
        targets (torch.Tensor or np.ndarray): Actual pressures.
        u_out (torch.Tensor or np.ndarray): Expiratory valve control signal.

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert to numpy if tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to ensure 1D alignment
    preds = preds.flatten()
    targets = targets.flatten()
    u_out = u_out.flatten()

    # Filter for inspiratory phase (u_out == 0)
    # Note: u_out is binary 0/1, but we use < 0.5 for float safety
    inspiratory_mask = u_out < 0.5

    if np.sum(inspiratory_mask) == 0:
        return 0.0

    valid_preds = preds[inspiratory_mask]
    valid_targets = targets[inspiratory_mask]

    # Calculate MAE
    mae = np.mean(np.abs(valid_preds - valid_targets))
    return mae
