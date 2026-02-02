import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python's random, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_metric(y_true, y_pred, seq_scored=68):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function slices the input arrays to the scored sequence length (seq_scored),
    aggregates errors globally across all samples and positions (to avoid batch-averaging bias),
    computes the RMSE for each of the 5 target columns, and returns the mean RMSE.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Expected shape: (N, seq_scored, 5) or (N, seq_len, 5).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Expected shape: (N, seq_len, 5) or (N, seq_scored, 5).
        seq_scored (int): The number of sequence positions to include in scoring.
            Defaults to 68.

    Returns:
        float: The MCRMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Slice predictions to the scored length if they exceed it
    # y_pred shape: (N, Length, 5)
    if y_pred.shape[1] > seq_scored:
        y_pred = y_pred[:, :seq_scored, :]

    # Slice ground truth to the scored length if they exceed it
    # y_true shape: (N, Length, 5)
    if y_true.shape[1] > seq_scored:
        y_true = y_true[:, :seq_scored, :]

    # Ensure shapes match after slicing
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch after slicing: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Flatten the batch and sequence dimensions to aggregate globally
    # New shape: (N * seq_scored, 5)
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate Mean Squared Error (MSE) for each column
    # axis=0 aggregates over all samples and positions
    mse = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate Root Mean Squared Error (RMSE) for each column
    rmse = np.sqrt(mse)

    # Calculate Mean Columnwise RMSE (MCRMSE)
    mcrmse = np.mean(rmse)

    return mcrmse
