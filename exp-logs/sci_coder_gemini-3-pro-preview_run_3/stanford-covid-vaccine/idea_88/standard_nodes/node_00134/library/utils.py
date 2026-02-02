import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_pred, y_true):
    """
    Calculates the MCRMSE loss for training (PyTorch version).

    According to the strategy, this loss is calculated on ALL 5 target columns
    to leverage Multi-Task Learning.

    Args:
        y_pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len, 5).
        y_true (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 5).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Slice to the scored sequence length (first 68 positions)
    # Shape becomes: (Batch, 68, 5)
    y_pred_scored = y_pred[:, : Config.SEQ_SCORED, :]
    y_true_scored = y_true[:, : Config.SEQ_SCORED, :]

    # Calculate MSE per element
    mse = (y_pred_scored - y_true_scored) ** 2

    # Average over batch and sequence length dimensions to get MSE per target column
    # We want the mean over (Batch, Seq_Len) for each of the 5 columns.
    # Shape becomes: (5,)
    mse_per_col = torch.mean(mse, dim=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # The final loss is the mean of the RMSEs across columns
    loss = torch.mean(rmse_per_col)

    return loss


def mcrmse_metric(y_pred, y_true):
    """
    Calculates the MCRMSE metric for validation (NumPy version).

    This metric is calculated ONLY on the 3 scored columns:
    - reactivity (index 0)
    - deg_Mg_pH10 (index 1)
    - deg_Mg_50C (index 3)

    Args:
        y_pred (np.ndarray): Predicted values of shape (N, Seq_Len, 5).
        y_true (np.ndarray): Ground truth values of shape (N, Seq_Len, 5).

    Returns:
        float: MCRMSE score.
    """
    # Slice to the scored sequence length
    y_pred_scored = y_pred[:, : Config.SEQ_SCORED, :]
    y_true_scored = y_true[:, : Config.SEQ_SCORED, :]

    # Define the indices of the columns that are actually scored
    # 0: reactivity
    # 1: deg_Mg_pH10
    # 2: deg_pH10 (Auxiliary)
    # 3: deg_Mg_50C
    # 4: deg_50C (Auxiliary)
    scored_indices = [0, 1, 3]

    # Filter for scored columns
    y_pred_filtered = y_pred_scored[:, :, scored_indices]
    y_true_filtered = y_true_scored[:, :, scored_indices]

    # Flatten the batch and sequence dimensions to compute global MSE per column
    # New shape: (N * 68, 3)
    y_pred_flat = y_pred_filtered.reshape(-1, len(scored_indices))
    y_true_flat = y_true_filtered.reshape(-1, len(scored_indices))

    # Compute MSE per column
    mse_per_col = np.mean((y_pred_flat - y_true_flat) ** 2, axis=0)

    # Compute RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Return the mean of the RMSEs
    return np.mean(rmse_per_col)


def save_checkpoint(state, is_best, checkpoint_dir=Config.WORKING_DIR):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint files.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the current checkpoint
    filename = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(state, filename)

    # If this is the best model, save a copy as best_model.pth
    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_filename)
