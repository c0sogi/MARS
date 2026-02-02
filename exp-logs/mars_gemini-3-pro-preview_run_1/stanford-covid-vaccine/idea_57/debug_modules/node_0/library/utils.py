import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric is calculated by:
    1. Computing the RMSE for each of the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    2. Taking the arithmetic mean of these column-wise RMSEs.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
                                             Shape should be (N, seq_scored, num_targets) or (N*seq_scored, num_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
                                             Shape must match y_true.

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in calculate_mcrmse: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Determine the number of target columns (last dimension)
    # Expected to be 3 based on Config.TARGET_COLS
    num_targets = y_true.shape[-1]

    column_rmses = []

    # Iterate over each target column
    for i in range(num_targets):
        # Extract the specific column and flatten to 1D array
        # This aggregates over batch and sequence length to get a single RMSE per target type
        y_true_col = y_true[..., i].flatten()
        y_pred_col = y_pred[..., i].flatten()

        # Calculate MSE for this column
        mse = np.mean((y_true_col - y_pred_col) ** 2)

        # Calculate RMSE for this column
        rmse = np.sqrt(mse)
        column_rmses.append(rmse)

    # MCRMSE is the mean of the column-wise RMSEs
    mcrmse = np.mean(column_rmses)

    return float(mcrmse)
