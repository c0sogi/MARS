import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def compute_mcrmse(preds, targets):
    """
    Computes Mean Columnwise Root Mean Squared Error (MCRMSE) strictly adhering to
    competition evaluation protocols:
    1. Slices data to the first 'seq_scored' positions (68).
    2. Filters columns to the specific scored targets.
    3. Computes RMSE per column and averages them.

    Args:
        preds: Predictions tensor or array of shape (Batch, Seq_Len, Channels).
        targets: Ground truth tensor or array of shape (Batch, Seq_Len, Channels).

    Returns:
        float: The computed MCRMSE score.
    """
    # Ensure inputs are torch tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # 1. Slice to scored sequence length (first 68 positions)
    # Note: Model outputs 107 positions, but only first 68 have ground truth for scoring.
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]
    targets_sliced = targets[:, : Config.SEQ_SCORED, :]

    # 2. Filter to scored columns
    # We only score on reactivity, deg_Mg_pH10, and deg_Mg_50C (indices 0, 1, 3)
    preds_filtered = preds_sliced[:, :, Config.SCORING_INDICES]
    targets_filtered = targets_sliced[:, :, Config.SCORING_INDICES]

    # 3. Compute MSE per column
    # Average over batch (dim 0) and sequence position (dim 1)
    mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

    # 4. Compute RMSE per column
    rmse = torch.sqrt(mse)

    # 5. Average RMSEs across the scored columns
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
