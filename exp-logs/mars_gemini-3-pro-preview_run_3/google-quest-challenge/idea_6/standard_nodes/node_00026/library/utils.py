import os
import random
import numpy as np
import torch
from scipy import stats
from library.config import Config


def seed_everything(seed=42):
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


def compute_spearmanr(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth target values of shape (N, 30).
        y_pred (np.ndarray or torch.Tensor): Predicted target values of shape (N, 30).

    Returns:
        float: The mean Spearman's rank correlation coefficient.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    correlations = []
    num_targets = y_true.shape[1]

    for i in range(num_targets):
        # Compute Spearman correlation for the current column
        # spearmanr returns a Result object or tuple (correlation, p-value)
        # We only need the correlation coefficient (index 0)
        # We use the slice to get 1D arrays
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        # Handle constant columns which cause undefined correlation (NaN)
        # Scipy handles this by returning NaN, which we filter out later with nanmean
        corr = stats.spearmanr(col_true, col_pred)[0]
        correlations.append(corr)

    # Return the mean of correlations, ignoring NaNs that might occur
    # if a column is constant (std=0)
    return float(np.nanmean(correlations))


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch.
        score (float): Validation score.
        filename (str): Name of the file to save (relative to Config.WORKING_DIR).
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }

    save_path = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, save_path)


def load_checkpoint(
    filename, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Name of the file to load (relative to Config.WORKING_DIR).
        model: The PyTorch model to load weights into.
        optimizer: (Optional) The optimizer to load state into.
        scheduler: (Optional) The scheduler to load state into.
        device: The device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch/score).
    """
    load_path = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
