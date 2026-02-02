import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for python, numpy, and pytorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(predictions, targets):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_targets).
        targets (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    # Convert tensors to numpy if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    num_cols = predictions.shape[1]
    correlations = []

    for i in range(num_cols):
        pred_col = predictions[:, i]
        target_col = targets[:, i]

        # Handle constant columns where correlation is undefined
        if np.std(pred_col) == 0 or np.std(target_col) == 0:
            corr = 0.0
        else:
            # spearmanr returns (correlation, pvalue) or an object with .statistic
            # Accessing via index 0 is compatible with both legacy and new scipy versions
            res = spearmanr(pred_col, target_col)
            try:
                corr = res.statistic
            except AttributeError:
                corr = res[0]

        correlations.append(corr)

    # Return the mean of the column-wise correlations
    return np.nanmean(correlations)
