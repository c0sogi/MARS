import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_structure_to_indices(structure: str) -> np.ndarray:
    """
    Parses a dot-bracket structure string into an array of paired indices.
    Used for Latent Spatial Mixing to identify 3D neighbors.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (len(structure),) where arr[i] is the index
                    of the base paired with i. If i is unpaired, arr[i] = -1.
    """
    n = len(structure)
    indices = np.full(n, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
            else:
                # In case of malformed structure (unbalanced closing), ignore or handle
                pass

    return indices


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Optimizes the metric directly as the training objective.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates MCRMSE between predictions and targets.

        Args:
            pred: Tensor of shape (Batch, Seq_Len, Num_Targets) or (N, Num_Targets)
            target: Tensor of shape (Batch, Seq_Len, Num_Targets) or (N, Num_Targets)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate squared errors element-wise
        loss = self.mse(pred, target)

        # Determine dimensions to average over for column-wise MSE
        # If input is (B, L, C), dim=(0, 1). If (N, C), dim=0.
        if pred.dim() == 3:
            dims = (0, 1)
        else:
            dims = (0,)

        # MSE per column
        column_mse = torch.mean(loss, dim=dims)

        # RMSE per column
        column_rmse = torch.sqrt(column_mse)

        # Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(column_rmse)

        return mcrmse


def mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Numpy implementation of MCRMSE for validation metrics.

    Args:
        y_true: Ground truth array of shape (N, Num_Targets) or (N, Seq_Len, Num_Targets)
        y_pred: Prediction array of shape (N, Num_Targets) or (N, Seq_Len, Num_Targets)

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate squared errors
    squared_diff = (y_true - y_pred) ** 2

    # Mean squared error per column
    if y_true.ndim == 3:
        # Average over batch and sequence length
        mse_per_col = np.mean(squared_diff, axis=(0, 1))
    else:
        # Average over samples (flattened)
        mse_per_col = np.mean(squared_diff, axis=0)

    # Root mean squared error per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Mean of RMSEs
    return float(np.mean(rmse_per_col))
