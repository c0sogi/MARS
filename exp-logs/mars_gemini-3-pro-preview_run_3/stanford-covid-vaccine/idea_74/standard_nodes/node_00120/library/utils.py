import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
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


def get_score(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the competition.

    Logic:
    1. Converts inputs to NumPy arrays if they are Tensors.
    2. Slices predictions to the first 68 positions (Config.pred_len).
    3. Selects only the scored columns defined in Config.scored_columns.
    4. Computes RMSE for each column and returns the mean.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values. Shape (N, 68, 5).
        y_pred (np.array or torch.Tensor): Predicted values. Shape (N, 107, 5) or (N, 68, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Slice predictions to match the scored sequence length (68)
    # y_true is expected to be already 68 in length based on dataset
    if y_pred.shape[1] > Config.pred_len:
        y_pred = y_pred[:, : Config.pred_len, :]

    # Identify indices of columns to score
    # Config.target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.scored_columns = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.target_columns) if col in Config.scored_columns
    ]

    # Filter arrays to scored columns only
    # Shape becomes (N, 68, 3)
    y_true_scored = y_true[:, :, scored_indices]
    y_pred_scored = y_pred[:, :, scored_indices]

    # Compute RMSE per column
    # We flatten the batch and sequence dimensions to treat every position as a sample
    # Axis 0: Batch, Axis 1: Sequence, Axis 2: Column
    # Mean over axes 0 and 1 gives MSE per column
    mse_per_column = np.mean((y_true_scored - y_pred_scored) ** 2, axis=(0, 1))
    rmse_per_column = np.sqrt(mse_per_column)

    # MCRMSE is the mean of the RMSEs
    mcrmse = np.mean(rmse_per_column)

    return mcrmse
