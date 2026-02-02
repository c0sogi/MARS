import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic behavior in CuDNN backends.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for consistent results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_scored_indices():
    """
    Returns the indices of the columns that are used for the competition metric.
    Based on Config.TARGET_COLS and Config.SCORED_COLS.

    Returns:
        list: A list of integers representing the indices of scored columns.
    """
    full_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    return [i for i, col in enumerate(full_cols) if col in scored_cols]


def mask_unscored_channels(tensor):
    """
    Zeros out the channels corresponding to unscored targets (deg_pH10, deg_50C)
    in the feedback tensor to prevent noise injection during recurrent steps.

    Args:
        tensor (torch.Tensor): Input tensor of shape (Batch, Seq, Channels) or (Batch, Channels).

    Returns:
        torch.Tensor: A new tensor with unscored channels set to zero.
    """
    # Identify unscored indices
    full_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    unscored_indices = [i for i, col in enumerate(full_cols) if col not in scored_cols]

    # Create a mask on the same device as the input tensor
    mask = torch.ones_like(tensor)

    # Zero out the specific channels
    # We use ellipsis (...) to handle both (B, L, C) and (B, C) shapes
    for idx in unscored_indices:
        mask[..., idx] = 0.0

    return tensor * mask


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the
    scored positions and scored columns.

    This function is designed to calculate the 'Correct Global RMSE' by operating
    on the full validation set arrays, avoiding the bias of averaging batch RMSEs.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values. Shape (N, Seq_Len, 5).
        y_pred (np.ndarray or torch.Tensor): Predicted values. Shape (N, Seq_Len, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check dimensions
    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays (N, Seq, Channels), got {y_true.shape} and {y_pred.shape}"
        )

    # 1. Slice to the scored sequence length (first 68 positions)
    # Positions > 68 are not scored in the competition metric.
    scored_len = Config.SCORED_LENGTH
    y_true_scored = y_true[:, :scored_len, :]
    y_pred_scored = y_pred[:, :scored_len, :]

    # 2. Select only the scored columns
    # We only score: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = get_scored_indices()

    y_true_filtered = y_true_scored[..., scored_indices]
    y_pred_filtered = y_pred_scored[..., scored_indices]

    # 3. Compute MSE per column
    # Average over samples (axis 0) and sequence positions (axis 1)
    # Result shape: (3,) corresponding to the 3 scored targets
    mse_per_col = np.mean((y_true_filtered - y_pred_filtered) ** 2, axis=(0, 1))

    # 4. Compute RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # 5. Compute Mean of RMSEs (MCRMSE)
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
