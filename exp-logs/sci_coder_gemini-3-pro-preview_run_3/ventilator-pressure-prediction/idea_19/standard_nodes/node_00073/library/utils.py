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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate torch device (cuda or cpu).

    Returns:
        torch.device: The available device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedL1Loss(nn.Module):
    """
    Calculates Mean Absolute Error (L1 Loss) strictly on the inspiratory phase.
    The inspiratory phase is defined where u_out == 0.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, preds, targets, u_out):
        """
        Computes the masked L1 loss.

        Args:
            preds (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Ground truth pressure values.
            u_out (torch.Tensor): Control input indicating phase (0=inspiratory, 1=expiratory).

        Returns:
            torch.Tensor: The mean absolute error over the inspiratory phase.
        """
        # Create a boolean mask where u_out is 0 (inspiratory phase)
        # We assume u_out has the same shape or is broadcastable to preds/targets
        mask = u_out == 0

        # Select elements that satisfy the mask
        # masked_select returns a flattened 1D tensor of selected elements
        preds_masked = torch.masked_select(preds, mask)
        targets_masked = torch.masked_select(targets, mask)

        # Handle edge case where mask selects nothing (unlikely in this dataset but safe to handle)
        if preds_masked.numel() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # Calculate Mean Absolute Error
        loss = torch.abs(preds_masked - targets_masked).mean()

        return loss
