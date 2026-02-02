import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that assigns different weights to the inspiratory and
    expiratory phases of the breath.

    Weights are retrieved from Config:
    - Inspiration (u_out=0): Config.LOSS_INSP_WEIGHT
    - Expiration (u_out=1): Config.LOSS_EXP_WEIGHT
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.w_insp = Config.LOSS_INSP_WEIGHT
        self.w_exp = Config.LOSS_EXP_WEIGHT

    def forward(self, preds, targets, u_out):
        """
        Calculates the weighted L1 loss.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth pressure.
            u_out (torch.Tensor): Binary control input indicating expiratory phase (0=insp, 1=exp).

        Returns:
            torch.Tensor: The scalar weighted mean absolute error.
        """
        # Ensure inputs are flattened to handle (Batch, Seq) or (Batch, Seq, 1) shapes consistently
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        u_out = u_out.reshape(-1)

        # Calculate absolute error
        abs_error = torch.abs(preds - targets)

        # Generate weight mask based on u_out
        # u_out is 0 for inspiration, 1 for expiration
        weights = (1 - u_out) * self.w_insp + u_out * self.w_exp

        # Apply weights
        weighted_error = abs_error * weights

        # Return mean loss
        return weighted_error.mean()


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) strictly for the inspiratory phase (u_out=0).
    This matches the competition metric.

    Args:
        preds (torch.Tensor or np.ndarray): Model predictions.
        targets (torch.Tensor or np.ndarray): Ground truth pressure.
        u_out (torch.Tensor or np.ndarray): Binary control input (0=insp, 1=exp).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert Tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    u_out = u_out.reshape(-1)

    # Filter for inspiratory phase (u_out == 0)
    # The competition metric only scores the inspiratory phase.
    insp_mask = u_out == 0

    # Safety check to avoid division by zero if mask is empty (unlikely in valid batches)
    if np.sum(insp_mask) == 0:
        return 0.0

    # Calculate MAE on filtered data
    mae = np.mean(np.abs(preds[insp_mask] - targets[insp_mask]))

    return float(mae)
