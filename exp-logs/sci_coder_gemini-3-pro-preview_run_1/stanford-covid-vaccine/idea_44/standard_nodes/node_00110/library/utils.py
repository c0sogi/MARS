import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for each target column separately and then
    returns the average of these RMSEs. This aligns with the competition metric
    and corrects for potential artifacts when averaging across positions.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Expected shape: (Batch, Seq_Len, Targets) or (Batch*Seq_Len, Targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Expected shape: (Batch, Seq_Len, Targets) or (Batch*Seq_Len, Targets).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert torch tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Flatten to 2D array: (Number of Observations, Number of Targets)
    # We assume the last dimension is the targets (channels)
    if y_true.ndim == 3:
        num_targets = y_true.shape[2]
        y_true = y_true.reshape(-1, num_targets)
        y_pred = y_pred.reshape(-1, num_targets)
    elif y_true.ndim == 1:
        # Handle 1D case (single target)
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    # Calculate MSE for each column (target)
    # axis=0 aggregates over all samples and sequence positions
    mse_per_col = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the column RMSEs
    score = np.mean(rmse_per_col)

    return float(score)
