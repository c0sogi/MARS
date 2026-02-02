import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_metric(preds: torch.Tensor, targets: torch.Tensor):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    specifically for the scored columns and positions.

    The metric is calculated on the first 68 positions (Config.PRED_LEN).
    The scored columns are:
    - reactivity (Index 0)
    - deg_Mg_pH10 (Index 1)
    - deg_Mg_50C (Index 3)

    Args:
        preds (torch.Tensor): Predicted values of shape (B, Seq_Len, 5).
        targets (torch.Tensor): Ground truth values of shape (B, Seq_Len, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are on the same device
    device = preds.device
    targets = targets.to(device)

    # Slice to the scored sequence length (first 68 positions)
    # Shape becomes: (B, 68, 5)
    preds_scored = preds[:, : Config.PRED_LEN, :]
    targets_scored = targets[:, : Config.PRED_LEN, :]

    # Calculate Mean Squared Error for each element
    mse = (preds_scored - targets_scored) ** 2

    # Average over batch and sequence dimensions to get MSE per column
    # Shape: (5,)
    mse_per_column = torch.mean(mse, dim=(0, 1))

    # Calculate RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Select the specific columns used for the competition metric
    # Indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = [0, 1, 3]
    selected_rmse = rmse_per_column[scored_indices]

    # Calculate the mean of the RMSEs for the selected columns
    mcrmse = torch.mean(selected_rmse)

    return mcrmse.item()
