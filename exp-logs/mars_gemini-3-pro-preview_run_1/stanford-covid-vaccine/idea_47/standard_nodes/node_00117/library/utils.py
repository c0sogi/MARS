import torch
import numpy as np
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets fixed random seeds for reproducibility by calling the Config.set_seed method.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each target column separately and then
    returns the average of these RMSEs. This is distinct from calculating the
    RMSE of the entire flattened array or averaging per-position errors.

    Args:
        y_true (torch.Tensor or array-like): Ground truth values.
                                             Shape should be (Batch, Seq_Len, Num_Targets)
                                             or (N, Num_Targets).
        y_pred (torch.Tensor or array-like): Predicted values.
                                             Shape must match y_true.

    Returns:
        torch.Tensor: The scalar MCRMSE value.
    """
    # Ensure inputs are torch tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Ensure inputs are on the same device
    if y_true.device != y_pred.device:
        y_pred = y_pred.to(y_true.device)

    # Flatten the batch and sequence dimensions to (N_samples, Num_Targets)
    # This ensures we calculate metrics per target column across the entire batch/sequence
    num_targets = y_true.shape[-1]
    y_true_flat = y_true.view(-1, num_targets)
    y_pred_flat = y_pred.view(-1, num_targets)

    # 1. Compute Mean Squared Error (MSE) for each column independently
    # dim=0 aggregates over the samples, leaving one value per target column
    mse_per_column = torch.mean((y_true_flat - y_pred_flat) ** 2, dim=0)

    # 2. Compute RMSE for each column
    rmse_per_column = torch.sqrt(mse_per_column)

    # 3. Compute the average of the column-wise RMSEs
    mcrmse_val = torch.mean(rmse_per_column)

    return mcrmse_val
