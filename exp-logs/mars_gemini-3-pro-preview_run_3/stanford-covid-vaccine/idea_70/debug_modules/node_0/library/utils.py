import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.
    This serves as the objective function during training, optimizing on all 5 target columns.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, preds, targets):
        """
        Computes the loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets).
                                  Typically Seq_Len is 107.
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Scored, Num_Targets).
                                    Typically Seq_Scored is 68.

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Slice predictions to match the scored sequence length (68) provided in targets
        seq_scored = targets.shape[1]
        preds_sliced = preds[:, :seq_scored, :]

        # Calculate MSE for each column
        # We average over the batch (dim 0) and sequence (dim 1) dimensions
        # Resulting shape: (Num_Targets,)
        col_mse = torch.mean((preds_sliced - targets) ** 2, dim=(0, 1))

        # Calculate RMSE for each column
        col_rmse = torch.sqrt(col_mse)

        # Average RMSE across all columns to get MCRMSE
        loss = torch.mean(col_rmse)

        return loss


def compute_val_metric(preds, targets):
    """
    Computes the MCRMSE score specifically for the validation set logic.

    This function:
    1. Converts inputs to NumPy arrays if they are Tensors.
    2. Slices predictions to the first 68 positions (`seq_scored`).
    3. Filters both predictions and targets to keep only the 3 scored columns:
       ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'].
    4. Computes the MCRMSE on this subset.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions of shape (N, 107, 5) or (N, 68, 5).
        targets (np.ndarray or torch.Tensor): Ground truth of shape (N, 68, 5).

    Returns:
        float: The calculated MCRMSE score on the scored columns.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Slice predictions to seq_scored (68) if they cover the full length (107)
    if preds.shape[1] > Config.seq_scored:
        preds = preds[:, : Config.seq_scored, :]

    # Identify indices of the columns that are actually scored
    # Config.target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices should correspond to [0, 1, 3]
    scored_indices = [
        i for i, col in enumerate(Config.target_cols) if col in Config.scored_cols
    ]

    # Filter for scored columns
    preds_scored = preds[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # Calculate MSE per column (averaging over batch and sequence dimensions)
    # axis=(0, 1) flattens the N samples and 68 positions
    mse_per_col = np.mean((preds_scored - targets_scored) ** 2, axis=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
