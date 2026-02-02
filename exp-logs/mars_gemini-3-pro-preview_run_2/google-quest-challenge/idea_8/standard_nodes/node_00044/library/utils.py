import os
import random
import copy
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels, shape (N, num_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities, shape (N, num_targets).

    Returns:
        float: The mean Spearman correlation across all target columns.
    """
    # Ensure inputs are numpy arrays
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_targets = y_true.shape[1]
    spearman_scores = []

    for col in range(num_targets):
        t = y_true[:, col]
        p = y_pred[:, col]

        # Handle constant columns which result in undefined correlation (NaN)
        # We treat these as 0 correlation for stability, or let scipy handle it.
        # Scipy's spearmanr returns a result object, index 0 is correlation.
        if np.std(t) == 0 or np.std(p) == 0:
            corr = 0.0
        else:
            try:
                corr = spearmanr(t, p)[0]
            except Exception:
                corr = 0.0

        if np.isnan(corr):
            corr = 0.0

        spearman_scores.append(corr)

    return np.mean(spearman_scores)


def save_checkpoint(model, path):
    """
    Saves the model state dict to the specified path.
    Uses copy.deepcopy to prevent reference issues.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the checkpoint.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Deep copy the state dict to strictly avoid reference bugs
    state_dict = copy.deepcopy(model.state_dict())

    # Save to file
    torch.save(state_dict, path)
