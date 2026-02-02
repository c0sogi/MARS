import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MaskedL1Loss(nn.Module):
    """
    Custom L1 Loss (MAE) that only calculates loss for the inspiratory phase.
    The expiratory phase (u_out == 1) is ignored in the loss calculation.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target, u_out):
        """
        Calculates the masked L1 loss.

        Args:
            pred (torch.Tensor): Predicted pressures.
            target (torch.Tensor): Actual pressures.
            u_out (torch.Tensor): Control input for the exploratory valve (0 or 1).

        Returns:
            torch.Tensor: The mean absolute error calculated only where u_out == 0.
        """
        # Ensure inputs are flattened or compatible shapes
        if pred.shape != target.shape:
            pred = pred.view_as(target)

        if u_out.shape != target.shape:
            u_out = u_out.view_as(target)

        # Create mask: 1 where u_out is 0 (inspiratory), 0 otherwise
        # u_out is binary (0 or 1), so (1 - u_out) works, or (u_out == 0)
        mask = (u_out == 0).float()

        # Calculate element-wise L1 loss
        loss = self.l1(pred, target)

        # Apply mask
        masked_loss = loss * mask

        # Calculate mean over the valid elements
        # Sum of mask gives the number of valid elements
        mask_sum = mask.sum()

        # Avoid division by zero if batch has no inspiratory phase (unlikely but safe)
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class MetricTracker:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the tracker with a new value.

        Args:
            val (float): The value to update.
            n (int): The number of samples associated with this value (default 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
