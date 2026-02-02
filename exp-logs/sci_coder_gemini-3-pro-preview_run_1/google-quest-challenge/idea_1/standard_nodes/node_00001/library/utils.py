import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(preds, targets):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        preds: Numpy array or Torch Tensor of shape (n_samples, n_targets).
               Predicted probabilities in range [0, 1].
        targets: Numpy array or Torch Tensor of shape (n_samples, n_targets).
                 Ground truth labels in range [0, 1].

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs have the same shape
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    num_targets = preds.shape[1]
    correlations = []

    for i in range(num_targets):
        # Extract the i-th column for predictions and targets
        pred_col = preds[:, i]
        target_col = targets[:, i]

        # Compute Spearman correlation
        # spearmanr returns a result object or tuple; index 0 is the correlation coefficient
        corr = spearmanr(pred_col, target_col)[0]
        correlations.append(corr)

    # Return the mean correlation, handling potential NaNs gracefully
    return float(np.nanmean(correlations))
