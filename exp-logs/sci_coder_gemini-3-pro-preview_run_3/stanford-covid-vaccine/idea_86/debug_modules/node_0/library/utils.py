import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
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


def calculate_mcrmse(preds: torch.Tensor, targets: torch.Tensor):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    for the specific scored columns and sequence positions.

    Logic:
    1. Slices predictions to the first Config.SEQ_SCORED positions.
    2. Selects only the columns specified in Config.SCORED_COLS.
    3. Computes RMSE for each column.
    4. Returns the mean of these RMSEs.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets).
                              Usually (B, 107, 5).
        targets (torch.Tensor): Ground truth of shape (Batch, Seq_Scored, Num_Targets).
                                Usually (B, 68, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Move to CPU for metric calculation to avoid GPU memory overhead during validation
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # 1. Slice predictions to the scored sequence length
    # Targets are typically already length 68, but preds are length 107.
    seq_scored = Config.SEQ_SCORED
    preds_sliced = preds[:, :seq_scored, :]

    # Ensure targets match the sliced prediction shape in the sequence dimension
    # (Handle case where targets might be full length or already sliced)
    if targets.shape[1] > seq_scored:
        targets_sliced = targets[:, :seq_scored, :]
    else:
        targets_sliced = targets

    # 2. Identify indices of scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices should be [0, 1, 3]

    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS

    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # 3. Calculate RMSE for each scored column
    rmse_list = []

    for idx in scored_indices:
        # Select the specific column
        p_col = preds_sliced[:, :, idx]
        t_col = targets_sliced[:, :, idx]

        # Calculate MSE
        mse = torch.mean((p_col - t_col) ** 2)

        # Calculate RMSE
        rmse = torch.sqrt(mse)
        rmse_list.append(rmse)

    # 4. Calculate MCRMSE (Mean of RMSEs)
    mcrmse = torch.mean(torch.stack(rmse_list))

    return mcrmse.item()
