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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Logic:
    1. Slices predictions and targets to the scored sequence length (Config.SEQ_SCORED).
    2. Flattens the batch and sequence dimensions to perform global aggregation.
    3. Computes RMSE for each of the 5 target columns.
    4. Returns the mean of these RMSE values.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions of shape (B, SeqLen, 5).
        targets (torch.Tensor or np.ndarray): Ground truth of shape (B, SeqLen, 5) or (B, SeqScored, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert to torch tensors if inputs are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure inputs are on the same device (CPU is sufficient for metric calc)
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # Slice predictions to the scored length
    # Model output is likely (B, 107, 5), we need (B, 68, 5)
    if preds.shape[1] > Config.SEQ_SCORED:
        preds = preds[:, : Config.SEQ_SCORED, :]

    # Slice targets if they are padded to the full sequence length
    if targets.shape[1] > Config.SEQ_SCORED:
        targets = targets[:, : Config.SEQ_SCORED, :]

    # Verify shapes match after slicing
    assert (
        preds.shape == targets.shape
    ), f"Shape mismatch after slicing: Preds {preds.shape} vs Targets {targets.shape}"

    # Global Aggregation: Flatten batch and sequence dimensions
    # Shape becomes (N_total_scored_positions, 5)
    preds_flat = preds.reshape(-1, Config.NUM_TARGETS)
    targets_flat = targets.reshape(-1, Config.NUM_TARGETS)

    # Calculate MSE per column
    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
