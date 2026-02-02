import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Default is 42.
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


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the mean of the RMSE values for each column.
    MCRMSE = (1/N_cols) * sum_j( sqrt( (1/N_samples) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    This function assumes the last dimension of the input arrays corresponds to the
    different target columns (e.g., reactivity, deg_Mg_pH10, deg_Mg_50C).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
                                             Expected shape: (N_samples, ..., N_columns)
        y_pred (np.ndarray or torch.Tensor): Predicted values.
                                             Expected shape: (N_samples, ..., N_columns)

    Returns:
        float: The calculated MCRMSE value.
    """
    # Convert torch tensors to numpy arrays if necessary for calculation
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in mcrmse_loss: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Squared Error: (y - y_hat)^2
    sq_error = (y_true - y_pred) ** 2

    # Calculate MSE per column
    # We collapse all dimensions except the last one (which represents the columns/targets)
    # The last dimension is preserved to calculate RMSE for each target independently.
    reduce_axes = tuple(range(y_true.ndim - 1))
    mse_per_col = np.mean(sq_error, axis=reduce_axes)

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
