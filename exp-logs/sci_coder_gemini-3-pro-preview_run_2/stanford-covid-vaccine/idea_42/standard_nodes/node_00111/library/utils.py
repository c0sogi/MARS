import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
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


def get_device() -> torch.device:
    """
    Returns the available device (CUDA or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mcrmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    scored_indices: list = None,
) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the mean of the RMSE values for each scored column.

    Args:
        pred: Predicted tensor of shape (Batch, Seq_Len, Num_Targets).
        target: Ground truth tensor of shape (Batch, Seq_Len, Num_Targets).
        mask: Boolean or binary mask of shape (Batch, Seq_Len) indicating valid positions.
              If None, assumes all positions are valid.
        scored_indices: List of indices corresponding to the columns to be scored.
                        If None, uses Config.SCORED_TARGET_INDICES.

    Returns:
        Scalar tensor representing the MCRMSE loss.
    """
    if scored_indices is None:
        scored_indices = Config.SCORED_TARGET_INDICES

    # Select only the columns that contribute to the metric
    # shape: (Batch, Seq_Len, Num_Scored_Targets)
    pred_scored = pred[:, :, scored_indices]
    target_scored = target[:, :, scored_indices]

    # Calculate squared errors
    squared_diff = (pred_scored - target_scored) ** 2

    if mask is not None:
        # Expand mask to match the target dimensions: (Batch, Seq) -> (Batch, Seq, 1)
        # This allows broadcasting across the target columns
        mask_expanded = mask.unsqueeze(-1)

        # Apply mask to zero out invalid positions
        squared_diff = squared_diff * mask_expanded

        # Calculate Mean Squared Error per column
        # Use the actual number of valid elements as the denominator
        # Cite debug_lesson_8: Mask or Slice Padded Regions Before Computing Sequence Metrics

        # mask is (Batch, Seq), so mask.sum() is the total count of valid positions in the batch
        valid_elements = mask.sum()

        # Sum over Batch (0) and Sequence (1) dimensions, keeping Columns separate
        mse_per_column = squared_diff.sum(dim=(0, 1)) / (valid_elements + 1e-8)
    else:
        # If no mask, simple mean over batch and sequence dimensions
        mse_per_column = squared_diff.mean(dim=(0, 1))

    # Calculate RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = rmse_per_column.mean()

    return mcrmse
