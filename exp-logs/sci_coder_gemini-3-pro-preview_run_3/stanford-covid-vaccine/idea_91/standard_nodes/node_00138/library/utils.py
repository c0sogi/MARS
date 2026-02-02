import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_pred, y_true):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.
    This function is differentiable and intended for use during training.

    The metric is defined as the mean of the RMSE values calculated for each
    target column separately.

    Args:
        y_pred (torch.Tensor): Predicted values. Shape (N, L, C) or (N, C).
        y_true (torch.Tensor): Ground truth values. Shape (N, L, C) or (N, C).

    Returns:
        torch.Tensor: The calculated MCRMSE loss (scalar).
    """
    # Check for shape mismatch
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch in MCRMSE loss: pred {y_pred.shape} vs true {y_true.shape}"
        )

    # Calculate Squared Error
    mse = (y_pred - y_true) ** 2

    # Calculate Mean Squared Error per column
    # If 3D (Batch, Seq, Channels), average over Batch and Seq (dims 0 and 1)
    # If 2D (Batch, Channels), average over Batch (dim 0)
    if y_pred.dim() == 3:
        mse_per_column = torch.mean(mse, dim=(0, 1))
    elif y_pred.dim() == 2:
        mse_per_column = torch.mean(mse, dim=0)
    else:
        raise ValueError(
            f"Unsupported tensor dimension for MCRMSE loss: {y_pred.dim()}"
        )

    # Calculate Root Mean Squared Error per column
    # Adding a small epsilon for numerical stability during backprop
    rmse_per_column = torch.sqrt(mse_per_column + 1e-8)

    # Calculate Mean of RMSEs across columns
    loss = torch.mean(rmse_per_column)

    return loss


def compute_score(y_pred, y_true):
    """
    Computes the competition metric (MCRMSE) for validation.

    This function implements the specific scoring logic required by the competition:
    1. Slices the predictions and targets to the first `Config.SEQ_SCORED` (68) positions.
    2. Filters the data to include only the columns specified in `Config.SCORED_COLS`.
    3. Computes the MCRMSE on this subset.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted values. Shape (N, 107, 5).
        y_true (torch.Tensor or np.ndarray): Ground truth values. Shape (N, 68, 5) or (N, 107, 5).

    Returns:
        float: The computed MCRMSE score.
    """
    # Convert inputs to torch Tensors if they are numpy arrays
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Ensure tensors are on CPU for metric calculation and detached from graph
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()

    # 1. Slice to scored sequence length (Config.SEQ_SCORED = 68)
    seq_scored = Config.SEQ_SCORED

    # Slice both tensors. y_true might already be 68, but slicing [:68] is safe.
    # y_pred is typically 107.
    pred_scored = y_pred[:, :seq_scored, :]
    true_scored = y_true[:, :seq_scored, :]

    # 2. Filter for scored columns
    # We map the column names to indices based on Config.TARGET_COLS
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS

    # Find indices of scored columns within the target columns
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    if not scored_indices:
        raise ValueError("No matching scored columns found in target columns.")

    pred_filtered = pred_scored[:, :, scored_indices]
    true_filtered = true_scored[:, :, scored_indices]

    # 3. Compute MCRMSE
    score = mcrmse_loss(pred_filtered, true_filtered)

    return score.item()
