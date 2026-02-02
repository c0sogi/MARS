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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MaskedL1Loss(nn.Module):
    """
    Custom Loss function that calculates L1 Loss (MAE) only for the inspiratory phase.
    The expiratory phase is indicated by u_out=1, which is ignored.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target, u_out):
        """
        Calculates the masked L1 loss.

        Args:
            pred (torch.Tensor): Predicted pressure values.
            target (torch.Tensor): Actual pressure values.
            u_out (torch.Tensor): Control input indicating expiratory phase (1) or inspiratory phase (0).

        Returns:
            torch.Tensor: Scalar loss value representing the mean absolute error over the inspiratory phase.
        """
        # Ensure u_out has the same shape as pred/target for broadcasting
        if u_out.shape != pred.shape:
            u_out = u_out.view_as(pred)

        # Create mask: 1 for inspiratory phase (u_out == 0), 0 for expiratory (u_out == 1)
        mask = 1 - u_out

        # Calculate element-wise L1 loss
        loss = self.l1(pred, target)

        # Apply mask
        masked_loss = loss * mask

        # Calculate mean over the valid (inspiratory) elements
        # We sum the masked loss and divide by the sum of the mask (count of inspiratory steps)
        # Add a small epsilon or check for zero to avoid division by zero (though unlikely in this dataset)
        mask_sum = mask.sum()

        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        else:
            return masked_loss.sum() * 0.0
