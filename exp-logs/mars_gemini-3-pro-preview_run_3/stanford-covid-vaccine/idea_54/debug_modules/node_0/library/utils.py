import torch
import torch.nn as nn
import numpy as np
from library.config import set_seed, PRED_LEN, ALL_TARGETS, SCORED_TARGETS


def compute_mcrmse(preds, targets, scoring_only=False):
    """
    Computes Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function enforces metric integrity by:
    1. Slicing predictions and targets to the first 68 positions (PRED_LEN).
    2. Optionally filtering for only the 3 scored columns for validation.

    Args:
        preds (torch.Tensor | np.ndarray): Predictions of shape (Batch, Seq_Len, 5).
        targets (torch.Tensor | np.ndarray): Ground truth of shape (Batch, Seq_Len, 5).
        scoring_only (bool): If True, calculates metric only on scored columns
                             (reactivity, deg_Mg_pH10, deg_Mg_50C).

    Returns:
        torch.Tensor: The scalar MCRMSE score.
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)

    # Slice to the scored sequence length (first 68 positions)
    # Ground truth is only provided for these positions.
    if preds.shape[1] > PRED_LEN:
        preds = preds[:, :PRED_LEN, :]
    if targets.shape[1] > PRED_LEN:
        targets = targets[:, :PRED_LEN, :]

    # Calculate MSE per column (averaging over batch and sequence dimensions)
    # shape: (5,)
    mse = torch.mean((preds - targets) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Filter for scored columns if requested (for Validation/Evaluation)
    if scoring_only:
        # Identify indices of scored targets within the all_targets list
        # SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Resulting indices: [0, 1, 3]
        scored_indices = [i for i, t in enumerate(ALL_TARGETS) if t in SCORED_TARGETS]
        rmse = rmse[scored_indices]

    # Return mean of RMSEs across the selected columns
    return torch.mean(rmse)


class MCRMSELoss(nn.Module):
    """
    Standard MCRMSE loss function for optimizing all 5 targets during training.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # Always compute on all targets (scoring_only=False) for training stability
        # and to utilize all available signal.
        return compute_mcrmse(preds, targets, scoring_only=False)
