import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    During training, this loss is calculated over all 5 target columns to maximize
    signal utilization, as specified in the strategy.
    """

    def __init__(self):
        super().__init__()
        self.config = Config()

    def forward(self, preds, targets):
        """
        Args:
            preds: Model predictions of shape (Batch, Seq_Len_Total, 5).
                   Typically (Batch, 107, 5).
            targets: Ground truth targets of shape (Batch, Seq_Len_Scored, 5).
                     Typically (Batch, 68, 5).

        Returns:
            loss: Scalar tensor representing the mean of column-wise RMSEs.
        """
        # Slice predictions to match the scored sequence length (68) provided in targets
        preds_sliced = preds[:, : self.config.pred_len, :]

        # Calculate Squared Error
        mse = (preds_sliced - targets) ** 2

        # Calculate MSE per column (averaging over batch and sequence dimensions)
        # dim=0 is batch, dim=1 is sequence position
        mse_per_col = torch.mean(mse, dim=(0, 1))

        # Calculate RMSE per column
        # Add epsilon for numerical stability to prevent NaN gradients if MSE is 0
        rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

        # Average RMSE across all 5 columns
        loss = torch.mean(rmse_per_col)

        return loss


def calculate_mcrmse(preds, targets):
    """
    Calculates the official competition metric: MCRMSE on scored columns only.

    Logic:
    1. Slice predictions to the first 68 positions (seq_scored).
    2. Filter columns to only include: 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'.
    3. Compute RMSE for each column globally (across all samples and positions).
    4. Return the mean of these specific RMSEs.

    Args:
        preds: Predictions (N, 107, 5) as numpy array or torch Tensor.
        targets: Ground Truth (N, 68, 5) as numpy array or torch Tensor.

    Returns:
        mcrmse: float, the calculated metric.
    """
    config = Config()

    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # 1. Slice predictions to seq_scored (68)
    preds_sliced = preds[:, : config.pred_len, :]

    # 2. Identify indices of the scored columns
    # config.target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # config.scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    all_cols = config.target_cols
    scored_cols_set = set(config.scored_cols)

    scored_indices = [i for i, col in enumerate(all_cols) if col in scored_cols_set]

    # Filter data to keep only the scored columns
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets[:, :, scored_indices]

    # 3. Compute RMSE per column globally
    # Flatten batch and sequence dimensions to compute global statistics
    preds_flat = preds_filtered.reshape(-1, len(scored_indices))
    targets_flat = targets_filtered.reshape(-1, len(scored_indices))

    # Mean Squared Error per column
    mse_per_col = np.mean((preds_flat - targets_flat) ** 2, axis=0)

    # Root Mean Squared Error per column
    rmse_per_col = np.sqrt(mse_per_col)

    # 4. Average the RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
