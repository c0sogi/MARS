import os
import random
import numpy as np
import torch


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score_columns(data):
    """
    Filters the input data to return only the columns used for scoring.
    The scored columns are: reactivity, deg_Mg_pH10, deg_Mg_50C.
    Indices: 0, 1, 3.

    Args:
        data (np.ndarray or torch.Tensor): Input data of shape (..., 5).

    Returns:
        Sliced data of shape (..., 3).
    """
    # Indices corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    target_indices = [0, 1, 3]

    if isinstance(data, np.ndarray):
        return data[..., target_indices]
    elif torch.is_tensor(data):
        return data[..., target_indices]
    else:
        raise TypeError("Input data must be a numpy array or torch tensor.")


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    MCRMSE = (1/Nt) * Sum_j( sqrt( (1/n) * Sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Calculate MSE per column (averaging over samples/sequence positions)
    # We assume the last dimension is the columns (targets)
    # Flatten all dimensions except the last one for calculation
    num_columns = y_true.shape[-1]
    y_true_flat = y_true.reshape(-1, num_columns)
    y_pred_flat = y_pred.reshape(-1, num_columns)

    # Calculate Mean Squared Error for each column
    mse = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmse = np.sqrt(mse)

    # Calculate Mean of RMSEs (Columnwise)
    mcrmse_score = np.mean(rmse)

    return float(mcrmse_score)
