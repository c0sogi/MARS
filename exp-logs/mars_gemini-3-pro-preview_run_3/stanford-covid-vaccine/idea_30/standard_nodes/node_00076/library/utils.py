import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MCRMSE(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function explicitly slices the inputs to the first `Config.PRED_LEN` (68) positions
    and aggregates errors globally over the entire provided dataset to avoid batch-size bias.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
                                             Shape: (N, Seq_Len, 5) or (N, 68, 5).
        y_pred (torch.Tensor or np.ndarray): Predicted values.
                                             Shape: (N, Seq_Len, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Explicitly slice to the scored sequence length (first 68 positions)
    # This handles cases where inputs are padded to 107 or already 68.
    y_true = y_true[:, : Config.PRED_LEN, :]
    y_pred = y_pred[:, : Config.PRED_LEN, :]

    # Flatten samples and sequence positions to aggregate globally
    # Shape becomes (N * 68, 5)
    y_true_flat = y_true.reshape(-1, Config.NUM_TARGETS)
    y_pred_flat = y_pred.reshape(-1, Config.NUM_TARGETS)

    # Calculate MSE per column
    mse = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate RMSE per column
    rmse = np.sqrt(mse)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = np.mean(rmse)

    return float(mcrmse)
