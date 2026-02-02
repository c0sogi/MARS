import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import MODEL_PARAMS


def set_seed(seed=42):
    """
    Sets the seed for random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate device (GPU if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    Calculates the RMSE for each specified target column separately, then averages them.
    Only considers positions where the mask is True.
    """

    def __init__(self, scored_indices=None):
        """
        Args:
            scored_indices (list of int, optional): Indices of the columns to calculate loss on.
                                                    Defaults to [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C).
        """
        super(MCRMSELoss, self).__init__()
        # Default to the competition scored columns:
        # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
        if scored_indices is None:
            self.scored_indices = [0, 1, 3]
        else:
            self.scored_indices = scored_indices

    def forward(self, preds, targets, mask):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (batch_size, seq_len, num_targets).
            targets (torch.Tensor): Ground truth of shape (batch_size, seq_len, num_targets).
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len).
                                 True indicates a position that should be scored.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Select only the columns we want to score
        preds_scored = preds[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # Apply the mask to flatten the tensors.
        # We select only the elements where mask is True.
        # mask shape: (batch, seq) -> expanded or used for indexing
        # Using boolean indexing flattens the batch and seq dimensions
        mask_bool = mask.bool()

        # active_preds shape: (num_valid_positions, num_scored_columns)
        active_preds = preds_scored[mask_bool]
        active_targets = targets_scored[mask_bool]

        # Calculate MSE per column (dim=0 is the flattened batch*seq dimension)
        mse_per_col = torch.mean((active_preds - active_targets) ** 2, dim=0)

        # Calculate RMSE per column
        # Add a small epsilon to avoid NaN gradients if MSE is exactly 0
        rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

        # Average the RMSEs across the columns
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse
