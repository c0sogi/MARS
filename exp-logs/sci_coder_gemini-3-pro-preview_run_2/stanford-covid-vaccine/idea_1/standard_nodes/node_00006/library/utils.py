import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=42):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This loss function specifically evaluates the first `Config.SCORED_LEN` (68)
    positions of the sequence, as defined in the competition metric.
    """

    def __init__(self, target_indices=None):
        super(MCRMSELoss, self).__init__()
        self.scored_len = Config.SCORED_LEN
        self.target_indices = target_indices

    def forward(self, inputs, targets):
        """
        Computes the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Num_Targets)
                                    or (Batch, Scored_Len, Num_Targets).

        Returns:
            torch.Tensor: A scalar tensor containing the MCRMSE loss.
        """
        # Slice inputs to the scored length (e.g., first 68 positions)
        # inputs shape: (Batch, 107, 5) -> (Batch, 68, 5)
        inputs_scored = inputs[:, : self.scored_len, :]

        # Slice targets to the scored length
        # Handles cases where targets are padded to 107 or already 68
        targets_scored = targets[:, : self.scored_len, :]

        # Filter specific target columns if indices are provided
        if self.target_indices is not None:
            inputs_scored = inputs_scored[:, :, self.target_indices]
            targets_scored = targets_scored[:, :, self.target_indices]

        # Calculate Squared Error
        squared_diff = (inputs_scored - targets_scored) ** 2

        # Calculate MSE per column (averaging over batch and sequence dimensions)
        # Result shape: (Num_Targets,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate Mean of RMSEs (MCRMSE) across all columns
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse
