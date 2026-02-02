import os
import random
import numpy as np
import torch
import torch.nn as nn
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
        # Deterministic algorithms ensure reproducibility but may reduce performance slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WeightedL1Loss(nn.Module):
    """
    Custom L1 Loss function that applies different weights to the inspiratory and
    expiratory phases of the breath cycle.

    Strategy:
        - Inspiratory phase (u_out=0): Weight = 1.0 (Primary metric focus)
        - Expiratory phase (u_out=1): Weight = 0.1 (Maintain temporal context)
    """

    def __init__(self):
        super().__init__()
        self.inspiratory_weight = Config.INSPIRATORY_WEIGHT
        self.expiratory_weight = Config.EXPIRATORY_WEIGHT

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, u_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the weighted L1 loss.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth values.
            u_out (torch.Tensor): Binary control input (0 for inspiratory, 1 for expiratory).
                                  Must be broadcastable to preds/targets shape.

        Returns:
            torch.Tensor: The scalar weighted mean absolute error.
        """
        # Calculate element-wise absolute errors
        abs_errors = torch.abs(preds - targets)

        # Create weight mask based on u_out
        # If u_out == 0 (Inspiratory), weight is self.inspiratory_weight
        # If u_out == 1 (Expiratory), weight is self.expiratory_weight
        weights = (1 - u_out) * self.inspiratory_weight + u_out * self.expiratory_weight

        # Apply weights to the errors
        weighted_errors = abs_errors * weights

        # Return the mean loss
        return weighted_errors.mean()
