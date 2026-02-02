import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MaskedL1Loss(nn.Module):
    """
    Custom Loss function that calculates Mean Absolute Error (L1)
    only for the inspiratory phase (where u_out == 0).
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, prediction, target, u_out):
        """
        Calculates the masked L1 loss.

        Args:
            prediction (torch.Tensor): The predicted pressure values.
            target (torch.Tensor): The ground truth pressure values.
            u_out (torch.Tensor): The control input indicating the phase
                                  (0 for inspiratory, 1 for expiratory).

        Returns:
            torch.Tensor: The calculated masked mean absolute error.
        """
        # Calculate element-wise L1 loss
        # prediction and target shape: (Batch, Seq_Len) or (Batch, Seq_Len, 1)
        loss = self.l1(prediction, target)

        # Create mask: 1 where u_out is 0 (inspiratory), 0 otherwise
        # u_out might be int or float, so we ensure mask is float for multiplication
        mask = (1 - u_out).float()

        # Ensure mask shape matches loss shape if necessary (e.g. broadcasting)
        if mask.shape != loss.shape:
            mask = mask.view_as(loss)

        # Apply mask to the loss
        masked_loss = loss * mask

        # Calculate the number of valid (inspiratory) time steps
        # Add epsilon to avoid division by zero
        valid_count = mask.sum()

        if valid_count < 1e-8:
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        # Return the mean loss over the valid time steps
        return masked_loss.sum() / valid_count
