import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_true, y_pred, num_scored=Config.SCORED_LEN):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for validation scoring.

    This function handles both 2D (flattened) and 3D (sequence) inputs. If 3D inputs are provided,
    it slices the sequence to the first `num_scored` positions before flattening.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Shape: (N, L, C) or (N*L, C).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Shape: same as y_true.
        num_scored (int): Number of positions to score from the beginning of the sequence.
            Only applies if input is 3D (N, L, C). Defaults to Config.SCORED_LEN.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert tensors to numpy if needed to ensure consistent calculation on CPU
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle 3D shape (Batch, Seq_Len, Channels)
    if y_true.ndim == 3:
        # Slice to the scored sequence length if specified
        if num_scored is not None and num_scored < y_true.shape[1]:
            y_true = y_true[:, :num_scored, :]
            y_pred = y_pred[:, :num_scored, :]

        # Flatten to (Samples * Scored_Len, Channels)
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate Mean Squared Error (MSE) per column (target)
    # axis=0 averages over the samples/positions
    mse = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate Root Mean Squared Error (RMSE) per column
    rmse = np.sqrt(mse)

    # Calculate the Mean of the column-wise RMSEs
    mcrmse = np.mean(rmse)

    return float(mcrmse)
