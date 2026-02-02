import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightedL1Loss(nn.Module):
    """
    Implements a Weighted L1 Loss function.

    Assigns different weights to the inspiratory and expiratory phases of the breath
    to focus model capacity on the scored metric while maintaining temporal stability.
    """

    def __init__(self):
        super().__init__()
        self.insp_weight = Config.INSPIRATORY_WEIGHT
        self.exp_weight = Config.EXPIRATORY_WEIGHT
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, targets, u_out):
        """
        Calculates the weighted L1 loss.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth values.
            u_out (torch.Tensor): Control input indicating phase (0=Inspiratory, 1=Expiratory).

        Returns:
            torch.Tensor: Scalar loss value (mean of weighted element-wise losses).
        """
        # Calculate raw element-wise L1 loss
        loss_unreduced = self.l1(preds, targets)

        # Create weight mask based on u_out
        # u_out is 0 for inspiratory, 1 for expiratory
        # weights = (1 - u_out) * w_insp + u_out * w_exp
        weights = (1 - u_out) * self.insp_weight + u_out * self.exp_weight

        # Apply weights
        weighted_loss = loss_unreduced * weights

        # Return mean loss
        return weighted_loss.mean()


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) for the inspiratory phase only.

    The competition metric only scores the inspiratory phase (u_out == 0).

    Args:
        preds (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.
        u_out (torch.Tensor): Control input indicating phase (0=Inspiratory, 1=Expiratory).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Create a boolean mask for the inspiratory phase (u_out == 0)
    # Ensure u_out is treated as a comparison compatible type
    mask = u_out == 0

    # Filter predictions and targets
    insp_preds = preds[mask]
    insp_targets = targets[mask]

    # Check if there are any inspiratory samples to avoid division by zero
    if insp_preds.numel() == 0:
        return 0.0

    # Calculate MAE
    mae = torch.abs(insp_preds - insp_targets).mean()

    return mae.item()
