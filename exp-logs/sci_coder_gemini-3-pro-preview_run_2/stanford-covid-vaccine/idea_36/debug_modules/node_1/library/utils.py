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


def mcrmse_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, scored_indices: list = None
) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each specified column separately and then
    takes the average of those RMSEs. It handles inputs of arbitrary leading
    dimensions (e.g., (Batch, Channels) or (Batch, Seq, Channels)) by flattening
    them into a single sample dimension.

    Args:
        y_pred (torch.Tensor): Predicted values. Last dimension must be channels.
        y_true (torch.Tensor): Ground truth values. Last dimension must be channels.
        scored_indices (list, optional): List of channel indices to include in the metric.
                                         Defaults to Config.SCORED_TARGET_INDICES.

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    if scored_indices is None:
        scored_indices = Config.SCORED_TARGET_INDICES

    # Ensure inputs are float tensors
    y_pred = y_pred.float()
    y_true = y_true.float()

    # Flatten all dimensions except the last (channel) dimension
    # This handles shapes like (Batch, SeqLen, Channels) -> (Batch*SeqLen, Channels)
    num_channels = y_pred.shape[-1]
    y_pred_flat = y_pred.view(-1, num_channels)
    y_true_flat = y_true.view(-1, num_channels)

    # Select only the columns that are being scored
    y_pred_scored = y_pred_flat[:, scored_indices]
    y_true_scored = y_true_flat[:, scored_indices]

    # Calculate MSE for each column (averaging over the sample dimension)
    mse = torch.mean((y_pred_scored - y_true_scored) ** 2, dim=0)

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate the mean of the column-wise RMSEs
    loss = torch.mean(rmse)

    return loss
