import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_pred, y_true):
    """
    Calculates the MCRMSE (Mean Columnwise Root Mean Squared Error) loss using PyTorch tensors.
    This metric averages the RMSE of each target column, correcting for the 'Mean of Sqrts' artifact.

    Args:
        y_pred (torch.Tensor): Predicted values. Shape (Batch, Seq_Len, Channels) or (N, Channels).
        y_true (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len, Channels) or (N, Channels).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Ensure float precision
    y_pred = y_pred.float()
    y_true = y_true.float()

    # Flatten to (N_samples * Seq_Len, Channels) if input is 3D
    # We assume the last dimension is the channel dimension
    if y_pred.dim() == 3:
        num_channels = y_pred.shape[-1]
        y_pred = y_pred.reshape(-1, num_channels)
        y_true = y_true.reshape(-1, num_channels)

    # Calculate MSE for each column (target)
    # dim=0 averages over the batch/sequence dimension
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate the average of the RMSEs across columns
    loss = torch.mean(rmse)

    return loss


def metric_mcrmse(y_true, y_pred):
    """
    Calculates the MCRMSE metric using NumPy arrays for evaluation.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The MCRMSE score.
    """
    # Convert to numpy arrays if they aren't already
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Flatten to (N_samples * Seq_Len, Channels) if input is 3D
    if y_true.ndim == 3:
        num_channels = y_true.shape[-1]
        y_true = y_true.reshape(-1, num_channels)
        y_pred = y_pred.reshape(-1, num_channels)

    # Calculate MSE for each column
    # axis=0 averages over the batch/sequence dimension
    mse = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse = np.sqrt(mse)

    # Calculate the average of the RMSEs
    score = np.mean(rmse)

    return float(score)
