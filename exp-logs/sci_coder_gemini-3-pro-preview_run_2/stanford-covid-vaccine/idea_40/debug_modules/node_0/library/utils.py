import os
import random
import ast
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the PyTorch device to use (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_list_column(x):
    """
    Parses a string representation of a list (e.g., '[0.1, 0.2, ...]') into a NumPy array.
    Used for processing target columns in the CSV metadata.

    Args:
        x (str or list): The input value to parse.

    Returns:
        np.ndarray: A float32 numpy array. Returns an empty array on failure.
    """
    try:
        if isinstance(x, str):
            # Evaluate the string as a Python literal (list)
            val = ast.literal_eval(x)
            return np.array(val, dtype=np.float32)
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        return np.array([], dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def mcrmse(y_true, y_pred, scored_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    MCRMSE = (1/Nt) * Sum_j( sqrt( (1/n) * Sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Args:
        y_true (torch.Tensor): Ground truth values. Shape (Batch, SeqLen, Channels) or (N, Channels).
        y_pred (torch.Tensor): Predicted values. Shape (Batch, SeqLen, Channels) or (N, Channels).
        scored_indices (list[int], optional): Indices of the channels to include in the metric calculation.
                                              If None, all channels are used.

    Returns:
        torch.Tensor: The scalar MCRMSE value.
    """
    # Ensure inputs are floating point tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Flatten to (Total_Samples, Channels) if input is 3D (Batch, Seq, Channels)
    if y_true.dim() == 3:
        num_channels = y_true.shape[-1]
        y_true = y_true.reshape(-1, num_channels)
        y_pred = y_pred.reshape(-1, num_channels)

    # Filter for scored columns if indices are provided
    if scored_indices is not None:
        y_true = y_true[:, scored_indices]
        y_pred = y_pred[:, scored_indices]

    # Calculate MSE for each column (channel)
    # dim=0 averages over the samples/sequence positions
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate the mean of the RMSEs across columns
    score = torch.mean(rmse)

    return score
