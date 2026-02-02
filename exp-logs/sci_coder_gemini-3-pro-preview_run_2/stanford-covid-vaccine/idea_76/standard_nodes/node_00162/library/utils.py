import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the seed for random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_global_rmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    on the valid 68 positions and scored columns.

    This function expects the full validation set predictions and targets
    to be passed in, ensuring the RMSE is calculated globally rather than
    averaged over batches.

    Args:
        preds: Numpy array or Tensor of shape (N, 107, 5) or (N, 68, 5).
        targets: Numpy array or Tensor of shape (N, 107, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Slice to the scored sequence length (first 68 positions)
    seq_scored = Config.SEQ_SCORED

    # Handle case where preds might already be sliced or full length
    if preds.shape[1] >= seq_scored:
        preds_scored = preds[:, :seq_scored, :]
    else:
        # If preds is smaller than seq_scored, this is an error state for this specific metric
        raise ValueError(
            f"Predictions sequence length {preds.shape[1]} is less than required {seq_scored}"
        )

    targets_scored = targets[:, :seq_scored, :]

    # Filter for the scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_cols_indices = Config.SCORED_COLS_INDICES

    rmses = []
    for col_idx in scored_cols_indices:
        # Extract the specific column for all samples and all scored positions
        p = preds_scored[:, :, col_idx]
        t = targets_scored[:, :, col_idx]

        # Calculate MSE over the flattened array (all samples * all positions)
        # This aggregates the error globally
        mse = np.mean((p - t) ** 2)
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    # The final metric is the mean of the column-wise RMSEs
    return np.mean(rmses)
