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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSEs for each of the 3 scored columns:
    reactivity, deg_Mg_pH10, and deg_Mg_50C.

    Args:
        y_true (torch.Tensor): Ground truth values. Shape (Batch, ..., 3).
        y_pred (torch.Tensor): Predicted values. Shape (Batch, ..., 3).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Calculate MSE per column (averaging over all dimensions except the last channel dim)
    # The last dimension is assumed to be the 3 targets.
    dims_to_reduce = list(range(y_true.ndim - 1))
    mse = torch.mean((y_true - y_pred) ** 2, dim=dims_to_reduce)

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Average RMSEs across the columns to get MCRMSE
    mcrmse = torch.mean(rmse)

    return mcrmse


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint including state dicts and metadata.

    Args:
        model (torch.nn.Module): The model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        epoch (int): The current epoch.
        loss (float): The validation loss (MCRMSE) at this checkpoint.
        path (str): The file path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)
