import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function:
    1. Slices the input predictions to the first `seq_scored` positions (68).
    2. Computes the Root Mean Squared Error (RMSE) for each of the 5 target columns independently.
    3. Returns the average of these RMSE values.

    This aligns with the Multi-Task Learning strategy where all 5 columns are optimized
    to provide robust representations, even though only 3 are scored.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Channels).
                                   Typically (B, 107, 5).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Scored, Channels).
                                    Typically (B, 68, 5).

        Returns:
            torch.Tensor: A scalar tensor representing the MCRMSE loss.
        """
        # Slice inputs to match the length of the ground truth (68 positions)
        inputs_sliced = inputs[:, : self.seq_scored, :]

        # Compute Squared Error: (Batch, Seq_Scored, Channels)
        squared_diff = (inputs_sliced - targets) ** 2

        # Compute Mean Squared Error (MSE) per column
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Compute RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Compute the mean of RMSEs across all 5 columns
        loss = torch.mean(rmse_per_column)

        return loss


def compute_competition_metric(preds, targets):
    """
    Computes the MCRMSE metric specifically for the 3 scored columns:
    - reactivity
    - deg_Mg_pH10
    - deg_Mg_50C

    This function is intended for validation and evaluation, not for gradient optimization.
    It handles slicing and column selection automatically based on the Config.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions of shape (N, 107, 5).
        targets (torch.Tensor or np.ndarray): Ground truth of shape (N, 68, 5).

    Returns:
        float: The MCRMSE score calculated over the 3 scored columns.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure computation happens on CPU to avoid GPU memory overhead during validation
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    seq_scored = Config.SEQ_SCORED

    # Slice predictions to the valid scored length (first 68 bases)
    preds_sliced = preds[:, :seq_scored, :]

    # Retrieve column names
    all_targets = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS

    # Identify indices of the columns that are actually scored
    # Typically indices [0, 1, 3] for reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [i for i, col in enumerate(all_targets) if col in scored_cols]

    if not scored_indices:
        raise ValueError("No matching scored columns found in target columns.")

    # Select only the scored columns from predictions and targets
    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # Compute MSE per column over the entire provided dataset
    # Flatten Batch and Sequence dimensions: (N * 68, 3)
    preds_flat = preds_scored.reshape(-1, len(scored_indices))
    targets_flat = targets_scored.reshape(-1, len(scored_indices))

    # Mean over the flattened dimension (dim 0)
    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)

    # RMSE per column
    rmse = torch.sqrt(mse)

    # Final Metric: Mean of the RMSEs
    metric = torch.mean(rmse)

    return metric.item()
