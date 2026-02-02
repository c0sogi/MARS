import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MaskedMAELoss(nn.Module):
    """
    Computes the Mean Absolute Error (L1 Loss) strictly on the inspiratory phase.
    The expiratory phase (indicated by u_out == 1) is masked out and not scored.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, u_out):
        """
        Forward pass for the masked loss.

        Args:
            y_pred (torch.Tensor): Predicted pressure values.
            y_true (torch.Tensor): Ground truth pressure values.
            u_out (torch.Tensor): Control input indicating phase (0: inspiratory, 1: expiratory).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure u_out has the same shape as y_pred for correct broadcasting
        # This handles cases where y_pred might be (B, L, 1) and u_out is (B, L)
        if u_out.shape != y_pred.shape:
            u_out = u_out.reshape(y_pred.shape)

        # Create mask: 1 for inspiratory (u_out == 0), 0 for expiratory (u_out == 1)
        mask = 1 - u_out

        # Calculate absolute error
        error = torch.abs(y_pred - y_true)

        # Apply mask to the error
        masked_error = error * mask

        # Calculate mean error over the valid (inspiratory) steps
        # Add epsilon to denominator to prevent division by zero
        loss = masked_error.sum() / (mask.sum() + 1e-8)

        return loss
