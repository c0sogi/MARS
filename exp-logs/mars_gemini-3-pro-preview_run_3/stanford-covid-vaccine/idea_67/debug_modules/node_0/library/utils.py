import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MCRMSE(y_true, y_pred, scored_only: bool = True):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function slices the inputs to the scored sequence length (Config.PRED_LEN)
    and optionally filters to the specific scored columns before calculating the metric.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
                                             Shape: (Batch, Seq_Len, 5)
        y_pred (torch.Tensor or np.ndarray): Predicted values.
                                             Shape: (Batch, Seq_Len, 5)
        scored_only (bool): If True, calculates the metric only on the 3 scored
                            columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
                            If False, calculates on all 5 columns.
                            Defaults to True.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert numpy arrays to torch tensors if necessary
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Ensure predictions are on the same device as targets
    if y_pred.device != y_true.device:
        y_pred = y_pred.to(y_true.device)

    # Slice the sequence dimension to the scored length (typically 68)
    # The metric is only defined on the first 'seq_scored' positions.
    y_true = y_true[:, : Config.PRED_LEN, :]
    y_pred = y_pred[:, : Config.PRED_LEN, :]

    # Filter columns if required
    if scored_only:
        # The scored columns are: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Based on the dataset order:
        # 0: reactivity
        # 1: deg_Mg_pH10
        # 2: deg_pH10
        # 3: deg_Mg_50C
        # 4: deg_50C
        scored_indices = [0, 1, 3]
        y_true = y_true[:, :, scored_indices]
        y_pred = y_pred[:, :, scored_indices]

    # Calculate MSE per column
    # We average over the batch (dim 0) and sequence (dim 1) dimensions
    # effectively treating all positions in the batch as samples for that column.
    mse = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Calculate the mean of the RMSEs across columns
    score = torch.mean(rmse)

    return score.item()
