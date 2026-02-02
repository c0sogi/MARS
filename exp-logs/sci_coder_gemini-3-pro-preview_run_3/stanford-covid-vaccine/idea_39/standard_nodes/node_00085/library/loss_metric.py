import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    Calculates the RMSE for each target column separately and then averages them.
    Used as the training objective on all 5 target columns.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Classes).
                                   Usually Seq_Len is 107.
            targets (torch.Tensor): Ground truth of shape (Batch, Pred_Len, Num_Classes).
                                    Usually Pred_Len is 68.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Slice inputs to match target length (pred_len = 68)
        # targets shape is (B, 68, 5), inputs is (B, 107, 5)
        pred_len = targets.size(1)
        inputs_sliced = inputs[:, :pred_len, :]

        # 2. Flatten batch and sequence dimensions to compute global MSE per column
        # Shape becomes (Batch * Pred_Len, Num_Classes)
        inputs_flat = inputs_sliced.reshape(-1, inputs_sliced.size(-1))
        targets_flat = targets.reshape(-1, targets.size(-1))

        # 3. Compute MSE per column
        mse = torch.mean((inputs_flat - targets_flat) ** 2, dim=0)

        # 4. Compute RMSE per column (add epsilon for numerical stability)
        rmse = torch.sqrt(mse + 1e-8)

        # 5. Average RMSE across all columns
        loss = torch.mean(rmse)

        return loss


def compute_metric(predictions, targets, config: Config):
    """
    Computes the competition metric: MCRMSE on specific scored columns.

    Logic:
    1. Slice predictions to the first 68 positions.
    2. Filter for the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    3. Compute RMSE for each column over the entire dataset.
    4. Return the mean of these RMSEs.

    Args:
        predictions (np.ndarray or torch.Tensor): Model predictions (N, 107, 5).
        targets (np.ndarray or torch.Tensor): Ground truth (N, 68, 5).
        config (Config): Configuration object containing column definitions.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # 1. Slice to seq_scored (68)
    pred_len = config.pred_len
    preds_sliced = predictions[:, :pred_len, :]
    targets_sliced = targets[:, :pred_len, :]

    # 2. Identify indices of scored columns
    # target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(config.target_cols) if col in config.scored_cols
    ]

    # 3. Filter data to only scored columns
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # 4. Flatten all dimensions except columns to compute global RMSE
    # Shape: (N * 68, 3)
    preds_flat = preds_filtered.reshape(-1, len(scored_indices))
    targets_flat = targets_filtered.reshape(-1, len(scored_indices))

    # 5. Compute RMSE per column
    mse = np.mean((preds_flat - targets_flat) ** 2, axis=0)
    rmse = np.sqrt(mse)

    # 6. Average RMSEs
    mcrmse = np.mean(rmse)

    return float(mcrmse)
