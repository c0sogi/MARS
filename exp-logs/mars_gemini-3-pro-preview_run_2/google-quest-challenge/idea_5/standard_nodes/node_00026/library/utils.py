import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed: int = 42) -> None:
    """
    Fixes random seeds across Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(predictions, targets) -> float:
    """
    Calculates the mean column-wise Spearman's correlation coefficient.

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, 30).
        targets (np.ndarray or torch.Tensor): Target labels of shape (N, 30).

    Returns:
        float: The mean Spearman's correlation coefficient across all columns.
    """
    # Convert tensors to numpy if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Initialize list to store correlations
    corrs = []

    # Iterate over each column (target variable)
    num_cols = predictions.shape[1]
    for col_idx in range(num_cols):
        pred_col = predictions[:, col_idx]
        target_col = targets[:, col_idx]

        # Calculate Spearman's correlation
        # spearmanr returns a result object or tuple where the first element is the correlation
        corr = spearmanr(pred_col, target_col)[0]
        corrs.append(corr)

    # Return the mean correlation, ignoring NaNs that might occur if a column has zero variance
    return float(np.nanmean(corrs))


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """
    Safely serializes the model state dictionary to the specified path.

    Args:
        model (torch.nn.Module): The model instance to save.
        path (str): The file path where the model state dict will be saved.
    """
    # Create the directory if it does not exist
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the model state dictionary
    torch.save(model.state_dict(), path)
