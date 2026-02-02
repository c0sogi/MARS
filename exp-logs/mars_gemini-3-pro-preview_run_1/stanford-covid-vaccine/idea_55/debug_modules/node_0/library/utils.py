import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each target column separately and then
    takes the average across columns. It handles both numpy arrays and
    torch tensors.

    Args:
        y_true: Ground truth values. Shape (..., n_columns).
        y_pred: Predicted values. Shape (..., n_columns).

    Returns:
        float: The calculated MCRMSE value.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Verify shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Flatten all dimensions except the last one (columns/targets)
    # This treats (Batch, Seq_Len, Channels) as (Batch*Seq_Len, Channels)
    num_columns = y_true.shape[-1]
    y_true_flat = y_true.reshape(-1, num_columns)
    y_pred_flat = y_pred.reshape(-1, num_columns)

    # Calculate MSE for each column
    # axis=0 averages over the samples/positions
    mse_per_col = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the RMSEs (MCRMSE)
    mcrmse = np.mean(rmse_per_col)

    return mcrmse
