import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The random seed value.
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

    The metric is calculated by:
    1. Computing RMSE for each target column separately over all samples and positions.
    2. Averaging the RMSE values across the columns.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
                                           Shape: (N, seq_len, num_targets) or (N*seq_len, num_targets).
        y_pred (np.array or torch.Tensor): Predicted values.
                                           Shape: (N, seq_len, num_targets) or (N*seq_len, num_targets).

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Reshape to 2D array: (Total_Samples, Num_Targets)
    # This flattens the batch and sequence length dimensions
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
    if y_pred.ndim == 3:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate Mean Squared Error (MSE) for each column (axis 0 is samples)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate Root Mean Squared Error (RMSE) for each column
    rmse = np.sqrt(mse)

    # Calculate the average of the RMSEs across columns
    mcrmse = np.mean(rmse)

    return float(mcrmse)


def save_checkpoint(state, filename):
    """
    Saves the training checkpoint to the specified filename.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filename)
