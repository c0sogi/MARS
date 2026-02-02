import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Optimizes the standard MCRMSE on all 5 target columns as specified for the training objective.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds: (Batch, Seq_Len_Pred, 5) - Predicted values (usually 107 length)
            targets: (Batch, Seq_Len_Target, 5) - Ground truth values (usually 68 length)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Slice predictions to match the length of targets (seq_scored=68)
        seq_scored = targets.shape[1]
        preds_sliced = preds[:, :seq_scored, :]

        # Calculate MSE for each column (dim=2) averaging over batch and sequence (dim=0, 1)
        mse = torch.mean((preds_sliced - targets) ** 2, dim=(0, 1))

        # Calculate RMSE for each column
        rmse = torch.sqrt(mse)

        # Average RMSE across all 5 columns
        loss = torch.mean(rmse)

        return loss


def metric_calculator(preds, targets):
    """
    Calculates the competition metric: MCRMSE on the 3 scored columns.

    Args:
        preds: (N, 107, 5) - Predictions for the full sequence.
        targets: (N, 68, 5) - Ground truth for the scored sequence.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are tensors
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # 1. Slice predictions to the scored sequence length (68)
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]

    # 2. Filter for the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Indices in the 5-column target list:
    # reactivity: 0, deg_Mg_pH10: 1, deg_pH10: 2, deg_Mg_50C: 3, deg_50C: 4
    scored_indices = [0, 1, 3]

    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets[:, :, scored_indices]

    # 3. Compute MCRMSE
    # Calculate MSE per column over the entire dataset (N, 68)
    mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Average RMSE across the 3 scored columns
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
