import os
import random
import numpy as np
import torch
from library.config import Config


def seed_all(seed=Config.SEED):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can be slower, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the device to be used for training/inference.

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    return torch.device(Config.DEVICE)


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric explicitly averages the RMSE of each column (target),
    rather than averaging the RMSE of each position, correcting the
    'Mean of Sqrts' artifact.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
                                             Expected shape: (Batch, Seq_Scored, 3) or (Batch*Seq_Scored, 3).
        y_pred (torch.Tensor or np.ndarray): Predicted values.
                                             Expected shape: (Batch, Seq_Scored, 3) or (Batch*Seq_Scored, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Convert numpy arrays to torch tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Ensure inputs are on CPU and float for calculation
    y_true = y_true.detach().cpu().float()
    y_pred = y_pred.detach().cpu().float()

    # Calculate Squared Error
    squared_error = (y_true - y_pred) ** 2

    # Determine dimensions to reduce.
    # We want to reduce all dimensions except the last one (which represents the columns/targets).
    # If shape is (Batch, Seq, Channels), dim=2 is channels. Reduce (0, 1).
    # If shape is (N, Channels), dim=1 is channels. Reduce (0).
    dims_to_reduce = list(range(y_true.dim() - 1))

    # Compute MSE per column: Mean over samples and sequence positions
    mse_per_column = torch.mean(squared_error, dim=dims_to_reduce)

    # Compute RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Compute Mean of RMSEs (MCRMSE)
    mcrmse_val = torch.mean(rmse_per_column)

    return mcrmse_val.item()
