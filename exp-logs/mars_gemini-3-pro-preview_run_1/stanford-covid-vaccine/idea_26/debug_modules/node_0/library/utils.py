import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def MCRMSE(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Metric Definition:
    MCRMSE = (1/N_cols) * SUM_j( sqrt( (1/N_samples) * SUM_i( (y_ij - y_hat_ij)^2 ) ) )

    This function assumes y_true and y_pred are PyTorch tensors containing only
    the scored positions and columns.

    Args:
        y_true (torch.Tensor): Ground truth values. Shape (N, 3) or (Batch, Seq, 3).
        y_pred (torch.Tensor): Predicted values. Shape (N, 3) or (Batch, Seq, 3).

    Returns:
        torch.Tensor: A scalar tensor containing the MCRMSE score.
    """
    # Ensure inputs are floating point tensors
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Calculate Squared Error
    squared_error = (y_true - y_pred) ** 2

    # Calculate Mean Squared Error per column
    # We flatten the batch and sequence dimensions if they exist to treat them as samples
    if y_true.dim() == 3:
        # Shape: (Batch, Seq, Cols) -> Mean over Batch and Seq (dims 0 and 1)
        mse_per_col = torch.mean(squared_error, dim=(0, 1))
    else:
        # Shape: (N, Cols) -> Mean over samples (dim 0)
        mse_per_col = torch.mean(squared_error, dim=0)

    # Calculate RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse_per_col)

    return mcrmse
