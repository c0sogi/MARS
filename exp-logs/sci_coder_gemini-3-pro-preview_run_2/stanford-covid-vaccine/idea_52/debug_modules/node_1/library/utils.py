import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates the loss only on the scored columns and scored sequence positions.
    """

    def __init__(self):
        super().__init__()
        # Determine indices of scored columns based on Config
        self.scored_indices = [
            i
            for i, col in enumerate(Config.ALL_TARGET_COLS)
            if col in Config.TARGET_COLS
        ]
        self.seq_scored = Config.SCORED_SEQ_LENGTH

    def forward(self, preds, targets):
        """
        Args:
            preds: (Batch, Seq_Len, 5)
            targets: (Batch, Seq_Len, 5)
        Returns:
            mcrmse: Scalar tensor
        """
        # Select only the scored sequence positions (0 to 67)
        preds_scored = preds[:, : self.seq_scored, :]
        targets_scored = targets[:, : self.seq_scored, :]

        # Select only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # We use integer indexing for the last dimension
        preds_filtered = preds_scored[:, :, self.scored_indices]
        targets_filtered = targets_scored[:, :, self.scored_indices]

        # Calculate MSE per column
        mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

        # Calculate RMSE per column
        rmse = torch.sqrt(mse)

        # Mean of RMSEs
        mcrmse = torch.mean(rmse)

        return mcrmse


def get_global_rmse(all_preds, all_targets):
    """
    Calculates the global MCRMSE over the entire dataset to avoid batch-averaging bias.

    Args:
        all_preds: Numpy array of shape (N_samples, Seq_Len, 5)
        all_targets: Numpy array of shape (N_samples, Seq_Len, 5)

    Returns:
        float: The global MCRMSE score.
    """
    # Determine indices of scored columns
    scored_indices = [
        i for i, col in enumerate(Config.ALL_TARGET_COLS) if col in Config.TARGET_COLS
    ]
    seq_scored = Config.SCORED_SEQ_LENGTH

    # Slice data to scored region and columns
    # Shape becomes (N_samples, 68, 3)
    preds_filtered = all_preds[:, :seq_scored, scored_indices]
    targets_filtered = all_targets[:, :seq_scored, scored_indices]

    # Calculate squared errors
    squared_errors = (preds_filtered - targets_filtered) ** 2

    # Calculate MSE per column (averaging over samples and sequence positions)
    # Result is shape (3,)
    column_mse = np.mean(squared_errors, axis=(0, 1))

    # Calculate RMSE per column
    column_rmse = np.sqrt(column_mse)

    # Average RMSE across columns
    global_mcrmse = np.mean(column_rmse)

    return global_mcrmse
