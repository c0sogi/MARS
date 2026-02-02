import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify paired bases.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (len(structure),) where index i contains
                    the index of the base paired with i. Returns -1 if unpaired.
    """
    # Initialize mapping with -1 (indicating unpaired)
    mapping = np.full(len(structure), -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                open_idx = stack.pop()
                # Record the pair in both directions
                mapping[open_idx] = i
                mapping[i] = open_idx

    return mapping


def scored_mcrmse(y_pred, y_true):
    """
    Calculates the MCRMSE (Mean Columnwise Root Mean Squared Error) on the
    scored positions and columns, as defined by the competition metric.

    Args:
        y_pred (torch.Tensor): Predicted values. Shape (B, Seq_Len, 5) or (B, Seq_Scored, 5).
        y_true (torch.Tensor): Ground truth values. Shape (B, Seq_Len, 5) or (B, Seq_Scored, 5).

    Returns:
        torch.Tensor: Scalar MCRMSE value.
    """
    # Ensure inputs are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)

    # 1. Slice to scored sequence length
    # Only the first Config.SEQ_SCORED (68) positions are evaluated.
    if y_pred.shape[1] > Config.SEQ_SCORED:
        y_pred = y_pred[:, : Config.SEQ_SCORED, :]

    if y_true.shape[1] > Config.SEQ_SCORED:
        y_true = y_true[:, : Config.SEQ_SCORED, :]

    # 2. Filter to scored columns
    # We only evaluate on 'reactivity', 'deg_Mg_pH10', and 'deg_Mg_50C'.
    # We dynamically determine these indices from the Config to ensure consistency.
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS

    # Identify indices: e.g., [0, 1, 3]
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    y_pred_scored = y_pred[:, :, scored_indices]
    y_true_scored = y_true[:, :, scored_indices]

    # 3. Calculate Column-wise RMSE
    # First, compute MSE for each column over the batch and sequence dimensions.
    mse = torch.mean((y_true_scored - y_pred_scored) ** 2, dim=(0, 1))

    # Compute RMSE for each column
    rmse = torch.sqrt(mse)

    # 4. Calculate Mean of RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse
