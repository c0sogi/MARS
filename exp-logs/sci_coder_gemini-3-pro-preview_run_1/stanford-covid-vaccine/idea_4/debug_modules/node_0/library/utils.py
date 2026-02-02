import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mcrmse_metric(y_true, y_pred, scored_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
        y_pred (np.array or torch.Tensor): Predicted values.
        scored_indices (list, optional): List of column indices to include in the metric.
                                         If None, all columns are used.

    Returns:
        float: The calculated MCRMSE value.
    """
    # Detach and move to cpu/numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten sample and sequence dimensions, keep target dimension
    # Expected input shape is (Batch, Seq_Len, Targets) or (Batch, Targets)
    num_targets = y_true.shape[-1]
    y_true = y_true.reshape(-1, num_targets)
    y_pred = y_pred.reshape(-1, num_targets)

    # Select specific columns if requested
    if scored_indices is not None:
        y_true = y_true[:, scored_indices]
        y_pred = y_pred[:, scored_indices]

    # Calculate MSE for each column
    # (N, Targets) -> (Targets,)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse = np.sqrt(mse)

    # Average RMSE across columns
    mcrmse = np.mean(rmse)

    return mcrmse


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model state to a file. If is_best is True, also saves to the best model path.

    Args:
        state (dict): The state dictionary to save (model weights, optimizer, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The path to save the current checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filename)

    # If this is the best model, copy it to the designated best model path
    if is_best:
        best_path = Config.MODEL_SAVE_PATH
        shutil.copyfile(filename, best_path)


def load_checkpoint(model, optimizer=None, filename=None):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Path to the checkpoint file.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if filename is None or not os.path.exists(filename):
        # Return defaults if no checkpoint found
        return 0, float("inf")

    print(f"Loading checkpoint from {filename}")
    # Load on the configured device
    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided and present
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", float("inf"))

    return start_epoch, best_score
