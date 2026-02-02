import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
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


def compute_mcrmse(preds, targets):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored targets.

    According to competition rules:
    1. Predictions are sliced to the first 68 positions (Config.SEQ_SCORED).
    2. Only specific columns are scored: reactivity, deg_Mg_pH10, deg_Mg_50C.
       Indices corresponding to sample_submission: [0, 1, 3].

    The column order is assumed to be:
    0: reactivity
    1: deg_Mg_pH10
    2: deg_pH10
    3: deg_Mg_50C
    4: deg_50C

    Args:
        preds (torch.Tensor or np.ndarray): Predictions of shape (Batch, Seq_Len, 5).
        targets (torch.Tensor or np.ndarray): Ground truth of shape (Batch, Seq_Len_Targets, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert to tensor if numpy array
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure float type
    preds = preds.float()
    targets = targets.float()

    # 1. Slicing: Slice predictions to the scored sequence length
    # Preds shape: (B, 107, 5) -> (B, 68, 5)
    if preds.shape[1] > Config.SEQ_SCORED:
        preds = preds[:, : Config.SEQ_SCORED, :]

    # Slice targets if they are full length (B, 107, 5) -> (B, 68, 5)
    # Targets usually come as 68 from the loader, but we handle 107 just in case.
    if targets.shape[1] > Config.SEQ_SCORED:
        targets = targets[:, : Config.SEQ_SCORED, :]

    # Verify shapes match after slicing
    assert (
        preds.shape == targets.shape
    ), f"Shape mismatch after slicing: Preds {preds.shape}, Targets {targets.shape}"

    # 2. Column Filtering
    # Scored: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    preds_filtered = preds[:, :, scored_indices]
    targets_filtered = targets[:, :, scored_indices]

    # 3. Calculation
    # MSE per column: Mean over Batch (dim 0) and Sequence (dim 1)
    # Result shape: (3,)
    mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

    # RMSE per column
    rmse = torch.sqrt(mse)

    # Mean of RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
