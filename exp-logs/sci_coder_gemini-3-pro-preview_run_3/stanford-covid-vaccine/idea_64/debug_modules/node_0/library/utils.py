import os
import random
import numpy as np
import torch
from library.config import config


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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored columns.

    Logic:
    1. Slices predictions to the scored sequence length (SEQ_SCORED = 68).
    2. Selects only the columns that contribute to the score:
       - reactivity
       - deg_Mg_pH10
       - deg_Mg_50C
    3. Computes RMSE for each selected column.
    4. Returns the mean of these RMSEs.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions of shape (B, SeqLen, 5).
        targets (torch.Tensor or np.ndarray): Ground truth of shape (B, SeqScored, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert to numpy if tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Determine indices of scored columns
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(config.TARGET_COLS) if col in config.SCORED_COLS
    ]

    # Slice predictions to match target length (SEQ_SCORED)
    # Targets are usually (B, 68, 5), Preds are (B, 107, 5)
    seq_scored = config.SEQ_SCORED

    # Safety check for dimensions
    if preds.shape[1] > seq_scored:
        preds = preds[:, :seq_scored, :]

    # Ensure shapes match now
    assert (
        preds.shape == targets.shape
    ), f"Shape mismatch after slicing: Preds {preds.shape}, Targets {targets.shape}"

    # Filter for scored columns only
    preds_scored = preds[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # Calculate RMSE per column
    # Flatten batch and sequence dimensions for calculation: (B * SeqScored, NumScoredCols)
    # Or simply mean over (0, 1) axes after square

    mse = np.mean((targets_scored - preds_scored) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)

    # Calculate Mean of RMSEs
    mcrmse = np.mean(rmse)

    return mcrmse
