import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates the RMSE for each of the 5 target columns separately and averages them.
    Handles masking to ensure only valid positions (seq_scored) contribute to the loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, masks):
        """
        Args:
            preds: (Batch, Seq_Len, 5)
            targets: (Batch, Seq_Len, 5)
            masks: (Batch, Seq_Len) - 1.0 for valid positions, 0.0 otherwise.

        Returns:
            loss: scalar tensor
        """
        # Calculate squared difference
        diff = preds - targets
        squared_diff = diff**2

        # Apply mask
        # masks shape: (B, L) -> (B, L, 1) to broadcast over channels
        mask_expanded = masks.unsqueeze(-1)
        masked_squared_diff = squared_diff * mask_expanded

        # Sum of squared errors per column
        # Sum over Batch (0) and Sequence (1) dimensions
        sum_squared_errors = torch.sum(masked_squared_diff, dim=(0, 1))

        # Count number of valid elements
        # Total valid positions = sum(masks)
        # Each column has the same number of valid positions
        num_valid = torch.sum(masks)

        # Avoid division by zero
        num_valid = torch.clamp(num_valid, min=1.0)

        # MSE per column
        mse_per_column = sum_squared_errors / num_valid

        # RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column + 1e-8)

        # Mean of RMSEs across the 5 columns
        loss = torch.mean(rmse_per_column)

        return loss


def calculate_metric_mcrmse(preds, targets, masks):
    """
    Calculates the competition metric: MCRMSE on specific scored columns.

    Logic:
    1. Slice data to the first 68 positions (Config.PRED_LEN).
    2. Filter for the 3 scored columns defined in Config.SCORED_COLS.
    3. Compute RMSE for each column.
    4. Return the mean of these RMSEs.

    Args:
        preds: (Batch, Seq_Len, 5) - Torch tensor or Numpy array
        targets: (Batch, Seq_Len, 5) - Torch tensor or Numpy array
        masks: (Batch, Seq_Len) - Torch tensor or Numpy array (used for verification/count)

    Returns:
        metric: float
    """
    # Convert to numpy if tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    # 1. Identify indices of scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS

    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # 2. Slice to scored length (first 68 positions)
    # The mask should ideally be 1.0 for the first 68 positions and 0.0 otherwise.
    # We explicitly slice to ensure we only look at the required region.
    pred_len = Config.PRED_LEN

    preds_sliced = preds[:, :pred_len, :]
    targets_sliced = targets[:, :pred_len, :]

    # 3. Filter for scored columns
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # 4. Compute RMSE per column
    # Flatten batch and sequence dimensions for calculation
    # Shape becomes (N_samples * 68, 3)
    diff_sq = (preds_filtered - targets_filtered) ** 2

    # Mean over the flattened dimensions (Batch, Seq)
    mse_per_col = np.mean(diff_sq, axis=(0, 1))

    # RMSE
    rmse_per_col = np.sqrt(mse_per_col)

    # 5. Mean columnwise RMSE
    metric = np.mean(rmse_per_col)

    return float(metric)
