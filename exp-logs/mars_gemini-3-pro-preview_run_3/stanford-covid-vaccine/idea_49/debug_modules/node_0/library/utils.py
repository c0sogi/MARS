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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, scored_cols_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function handles:
    1. Slicing predictions to the scored sequence length (first 68 bases).
    2. Filtering specific target columns if indices are provided.
    3. Computing RMSE per column and averaging them.

    Args:
        y_true: Ground truth array/tensor. Shape (Batch, 68, 5) or (Batch, 107, 5).
        y_pred: Prediction array/tensor. Shape (Batch, 107, 5).
        scored_cols_indices (list, optional): List of column indices to include in the metric.
                                              If None, all columns are used.

    Returns:
        float: The calculated MCRMSE value.
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Move to CPU to ensure consistent float32 precision and avoid GPU memory overhead
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    # Slice predictions to the scored length defined in Config
    # y_pred is typically (Batch, 107, Targets), we need (Batch, 68, Targets)
    if y_pred.shape[1] > Config.SEQ_SCORED:
        y_pred = y_pred[:, : Config.SEQ_SCORED, :]

    # If y_true was provided with full length (107), slice it as well
    if y_true.shape[1] > Config.SEQ_SCORED:
        y_true = y_true[:, : Config.SEQ_SCORED, :]

    # Verify shapes match after slicing
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in calculate_metric: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Filter specific columns if indices are provided
    # e.g., for validation scoring we might only want columns [0, 1, 3]
    if scored_cols_indices is not None:
        y_true = y_true[:, :, scored_cols_indices]
        y_pred = y_pred[:, :, scored_cols_indices]

    # Calculate MSE for each column: Mean over Batch (dim 0) and Sequence (dim 1)
    # Resulting shape: (Num_Selected_Targets,)
    mse = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate average of RMSEs across columns (MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
