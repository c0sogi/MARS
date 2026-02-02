import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(preds, targets):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        preds (np.ndarray): Predicted probabilities of shape (N, 30).
        targets (np.ndarray): Ground truth labels of shape (N, 30).

    Returns:
        float: The mean Spearman's correlation coefficient across all columns.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    corrs = []
    # Iterate over each column (target variable)
    for i in range(targets.shape[1]):
        # spearmanr returns a result object or tuple where the first element is the correlation
        # We use index 0 to access the correlation coefficient
        corr = spearmanr(preds[:, i], targets[:, i])[0]
        corrs.append(corr)

    # Return the mean of the correlations
    return np.mean(corrs)
