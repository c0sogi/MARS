import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcrmse(preds, targets) -> float:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) for the competition metric.

    Logic:
    1. Slices predictions and targets to the first 68 positions (Config.SEQ_SCORED).
    2. Selects only the scored columns defined in Config.SCORED_INDICES.
    3. Computes the RMSE for each of these columns individually.
    4. Returns the mean of these RMSE values.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions with shape (Batch, Seq_Len, 5).
        targets (torch.Tensor or np.ndarray): Ground truth with shape (Batch, Seq_Len, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are sliced to the scored sequence length (first 68 bases)
    # Assumes shape format (Batch, Sequence, Channels)
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]
    targets_sliced = targets[:, : Config.SEQ_SCORED, :]

    # Select only the columns that are scored
    # Config.SCORED_INDICES corresponds to ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
    preds_selected = preds_sliced[:, :, Config.SCORED_INDICES]
    targets_selected = targets_sliced[:, :, Config.SCORED_INDICES]

    # Compute RMSE per column
    # Flatten batch and sequence dimensions to (N_total_positions, N_scored_columns)
    preds_flat = preds_selected.reshape(-1, len(Config.SCORED_INDICES))
    targets_flat = targets_selected.reshape(-1, len(Config.SCORED_INDICES))

    # Calculate Mean Squared Error for each column
    mse_per_col = np.mean((targets_flat - preds_flat) ** 2, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Final Metric: Mean of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
