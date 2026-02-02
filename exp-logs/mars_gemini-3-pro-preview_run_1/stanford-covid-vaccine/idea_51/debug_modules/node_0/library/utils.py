import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (cuda or cpu).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mcrmse(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_scored: int = Config.PRED_LEN
) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each target column separately (averaging over
    batch and sequence dimensions) and then takes the mean of these RMSE values.
    It automatically slices the inputs to the first `num_scored` positions.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (batch_size, seq_len, num_targets).
        y_pred (torch.Tensor): Predicted tensor of shape (batch_size, seq_len, num_targets).
        num_scored (int): The number of positions to score from the start of the sequence.
                          Defaults to Config.PRED_LEN (68).

    Returns:
        torch.Tensor: The scalar MCRMSE score.
    """
    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Slice to the scored sequence length
    # Assumes shape is (Batch, Seq, Channels)
    if y_pred.shape[1] > num_scored:
        y_pred = y_pred[:, :num_scored, :]
    if y_true.shape[1] > num_scored:
        y_true = y_true[:, :num_scored, :]

    # Calculate MSE per column (target)
    # Average over batch (dim 0) and sequence (dim 1)
    # This corresponds to (1/n) * sum((y - y_hat)^2) in the formula
    mse_per_column = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Average RMSE across all columns to get MCRMSE
    score = torch.mean(rmse_per_column)

    return score
