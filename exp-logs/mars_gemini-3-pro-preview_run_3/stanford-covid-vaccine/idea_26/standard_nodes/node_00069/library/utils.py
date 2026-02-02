import os
import random
import numpy as np
import torch
from library.config import SEQ_SCORED


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def parse_structure_to_indices(structure):
    """
    Parses a dot-bracket structure string into an adjacency index array.

    This function converts the string representation of secondary structure
    (e.g., "((..))") into an integer array where each position contains the
    index of its paired base. Unpaired bases are represented by -1.

    Args:
        structure (str): Dot-bracket notation string.

    Returns:
        np.ndarray: Array of shape (len(structure),) with dtype int32.
                    arr[i] == j implies base i is paired with base j.
                    arr[i] == -1 implies base i is unpaired.
    """
    n = len(structure)
    indices = np.full(n, -1, dtype=np.int32)
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


def MCRMSE(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Slicing inputs to the first SEQ_SCORED positions.
    2. Computing RMSE for each target column independently.
    3. Averaging the RMSE values across all columns.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
                                             Shape: (Batch, Seq_Len, Num_Targets)
        y_pred (torch.Tensor or np.ndarray): Predicted values.
                                             Shape: (Batch, Seq_Len, Num_Targets)

    Returns:
        torch.Tensor: Scalar MCRMSE value.
    """
    # Ensure inputs are torch tensors
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Slice to the scored sequence length (SEQ_SCORED = 68)
    # Assumes shape is (Batch, Seq_Len, Channels)
    y_true_sliced = y_true[:, :SEQ_SCORED, :]
    y_pred_sliced = y_pred[:, :SEQ_SCORED, :]

    # Calculate MSE for each column (target type)
    # Averaging over Batch (dim 0) and Sequence (dim 1)
    mse = torch.mean((y_true_sliced - y_pred_sliced) ** 2, dim=(0, 1))

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse
