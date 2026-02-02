import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_all(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MaskedMCRMSELoss(nn.Module):
    """
    Custom loss module that computes MCRMSE only on the scored columns.
    It ignores auxiliary targets to prevent negative transfer.
    """

    def __init__(self, scored_indices=None):
        super(MaskedMCRMSELoss, self).__init__()
        self.scored_indices = (
            scored_indices if scored_indices is not None else Config.SCORED_INDICES
        )

    def forward(self, preds, targets):
        """
        Args:
            preds: Predictions tensor of shape (Batch, Seq_Len, Num_Targets)
            targets: Ground truth tensor of shape (Batch, Seq_Len, Num_Targets)
        Returns:
            mcrmse: Scalar tensor representing the loss.
        """
        # Select only the columns that contribute to the competition metric
        preds_scored = preds[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # Compute Mean Squared Error (MSE) for each scored column
        # Averaging over Batch (dim 0) and Sequence Length (dim 1)
        mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # Compute RMSE for each column (adding epsilon for numerical stability)
        rmse = torch.sqrt(mse + 1e-8)

        # Compute the mean of RMSEs across the scored columns
        mcrmse = torch.mean(rmse)

        return mcrmse


class GlobalRMSETracker:
    """
    Accumulates Sum of Squared Errors (SSE) and counts across batches
    to compute the mathematically correct global RMSE at the end of an epoch.
    """

    def __init__(self, scored_indices=None):
        self.scored_indices = (
            scored_indices if scored_indices is not None else Config.SCORED_INDICES
        )
        self.reset()

    def reset(self):
        self.sse = None  # Sum of squared errors per column
        self.count = 0  # Total number of elements (pixels/bases) processed

    def update(self, preds, targets):
        """
        Updates the tracker with predictions and targets from a batch.
        Args:
            preds: Predictions tensor or array (Batch, Seq_Len, Num_Targets)
            targets: Ground truth tensor or array (Batch, Seq_Len, Num_Targets)
        """
        # Convert torch tensors to numpy arrays if necessary
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Filter for scored columns
        preds_scored = preds[..., self.scored_indices]
        targets_scored = targets[..., self.scored_indices]

        # Flatten the batch and sequence dimensions to treat all positions as samples
        # Shape: (N_samples, N_scored_cols)
        preds_flat = preds_scored.reshape(-1, len(self.scored_indices))
        targets_flat = targets_scored.reshape(-1, len(self.scored_indices))

        # Calculate squared errors
        squared_errors = (preds_flat - targets_flat) ** 2

        # Sum errors per column
        col_sse = np.sum(squared_errors, axis=0)

        # Update state
        if self.sse is None:
            self.sse = col_sse
        else:
            self.sse += col_sse

        self.count += preds_flat.shape[0]

    def compute(self):
        """
        Computes the MCRMSE over all accumulated data.
        Returns:
            float: The global MCRMSE value.
        """
        if self.count == 0:
            return 0.0

        # Mean Squared Error per column
        mse = self.sse / self.count

        # Root Mean Squared Error per column
        rmse = np.sqrt(mse)

        # Mean Columnwise RMSE
        mcrmse = np.mean(rmse)

        return mcrmse
