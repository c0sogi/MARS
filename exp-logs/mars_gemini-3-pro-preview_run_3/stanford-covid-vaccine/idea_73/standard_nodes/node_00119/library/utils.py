import torch
import torch.nn as nn
import numpy as np
from library.config import Hyperparameters


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility using the configuration utility.
    """
    Hyperparameters.set_seed(seed)


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.
    Computes the loss across all 5 target columns as required for the training objective.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Hyperparameters.SEQ_SCORED

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Length, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Scored, 5).

        Returns:
            torch.Tensor: The MCRMSE loss scalar.
        """
        # Slice predictions to the scored sequence length (first 68 positions)
        # targets are already length 68 based on dataset description
        preds_sliced = preds[:, : self.seq_scored, :]

        # Calculate Squared Error
        mse = (preds_sliced - targets) ** 2

        # Calculate Mean Squared Error per column (averaging over Batch and Sequence dimensions)
        mse_per_col = torch.mean(mse, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Calculate Mean of RMSEs across all 5 columns
        loss = torch.mean(rmse_per_col)

        return loss


def metric_mcrmse_scored(preds, targets):
    """
    Calculates the MCRMSE specifically for the 3 scored columns:
    reactivity (idx 0), deg_Mg_pH10 (idx 1), and deg_Mg_50C (idx 3).

    Args:
        preds (torch.Tensor or np.ndarray): Predictions (Batch, Seq_Length, 5).
        targets (torch.Tensor or np.ndarray): Ground truth (Batch, Seq_Scored, 5).

    Returns:
        float: The calculated metric.
    """
    seq_scored = Hyperparameters.SEQ_SCORED

    # Convert inputs to torch Tensors if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.tensor(targets)

    # Detach gradients if present
    if preds.requires_grad:
        preds = preds.detach()

    # Slice predictions to scored length
    preds_sliced = preds[:, :seq_scored, :]

    # Select the scored columns based on the dataset column order:
    # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    # Scored indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = [0, 1, 3]

    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # Calculate MSE per column
    mse = (preds_scored - targets_scored) ** 2
    mse_per_col = torch.mean(mse, dim=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # Average RMSEs
    score = torch.mean(rmse_per_col)

    return score.item()
