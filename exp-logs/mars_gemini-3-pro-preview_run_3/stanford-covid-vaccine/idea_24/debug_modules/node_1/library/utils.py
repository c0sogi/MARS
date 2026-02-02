import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).
    This function is differentiable and can be used as a loss function.

    Args:
        y_true (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Num_Targets).
        y_pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Num_Targets).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Calculate MSE per column: average over batch and sequence dimensions
    # Shape: (Num_Targets,)
    colwise_mse = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    colwise_rmse = torch.sqrt(colwise_mse)

    # Average RMSEs across columns to get MCRMSE
    return torch.mean(colwise_rmse)


def get_scored_metrics(y_true, y_pred):
    """
    Calculates the MCRMSE specifically for the 3 scored columns:
    reactivity, deg_Mg_pH10, deg_Mg_50C.

    Based on the dataset schema and submission format, the column order is:
    0: reactivity
    1: deg_Mg_pH10
    2: deg_pH10
    3: deg_Mg_50C
    4: deg_50C

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
        y_pred (torch.Tensor or np.ndarray): Predicted values.

    Returns:
        float: The MCRMSE score for the scored columns.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Indices for the scored columns
    # reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]

    # Ensure tensors are on the same device
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)

    # Select the specific columns
    y_true_scored = y_true[:, :, scored_indices]
    y_pred_scored = y_pred[:, :, scored_indices]

    # Calculate MCRMSE for the selected columns
    colwise_mse = torch.mean((y_true_scored - y_pred_scored) ** 2, dim=(0, 1))
    colwise_rmse = torch.sqrt(colwise_mse)
    score = torch.mean(colwise_rmse)

    return score.item()
