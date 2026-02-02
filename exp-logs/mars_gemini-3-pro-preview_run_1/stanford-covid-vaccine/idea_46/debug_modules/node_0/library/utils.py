import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric calculates the RMSE for each target column separately and then
    averages these RMSE values. This is distinct from a global RMSE and ensures
    that each target contributes equally to the final score, regardless of its
    variance magnitude.

    Formula:
        MCRMSE = (1/N_t) * sum_{j=1}^{N_t} sqrt( (1/n) * sum_{i=1}^{n} (y_{ij} - y_hat_{ij})^2 )

    Args:
        y_true (torch.Tensor): Ground truth tensor. Expected shape: (Batch, Seq_Len, Targets)
                               or (Batch, Targets).
        y_pred (torch.Tensor): Predicted tensor. Expected shape matches y_true.

    Returns:
        torch.Tensor: A scalar tensor containing the MCRMSE value.
    """
    # Ensure inputs are floating point tensors
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Determine the number of targets (last dimension)
    num_targets = y_true.shape[-1]

    # Reshape inputs to (-1, num_targets) to flatten batch and sequence dimensions
    # This aggregates all predictions for a specific target column together
    y_true_flat = y_true.reshape(-1, num_targets)
    y_pred_flat = y_pred.reshape(-1, num_targets)

    # Calculate Mean Squared Error (MSE) for each column
    # dim=0 aggregates over the flattened samples
    mse_per_column = torch.mean((y_true_flat - y_pred_flat) ** 2, dim=0)

    # Calculate Root Mean Squared Error (RMSE) for each column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Calculate the mean of the column-wise RMSEs
    mcrmse_val = torch.mean(rmse_per_column)

    return mcrmse_val
