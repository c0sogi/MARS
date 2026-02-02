import os
import random
import numpy as np
import torch
from scipy import stats
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(preds, targets):
    """
    Computes the Mean Column-wise Spearman's Correlation Coefficient.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, 30).
        targets (np.ndarray or torch.Tensor): Ground truth labels of shape (N, 30).

    Returns:
        float: The mean Spearman's correlation across all 30 target columns.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure shapes match
    assert (
        preds.shape == targets.shape
    ), f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"

    correlations = []
    num_targets = preds.shape[1]

    for i in range(num_targets):
        # Calculate Spearman correlation for the current column
        # spearmanr returns a significance result object, we access the statistic (correlation)
        # If the input is constant, spearmanr might return nan. We handle this by replacing nan with 0.
        try:
            col_preds = preds[:, i]
            col_targets = targets[:, i]

            # Check for constant values to avoid warnings/NaNs if possible,
            # though scipy handles some cases, explicit handling is safer for metrics
            if np.all(col_preds == col_preds[0]) or np.all(
                col_targets == col_targets[0]
            ):
                corr = 0.0
            else:
                corr = stats.spearmanr(col_preds, col_targets).statistic

            if np.isnan(corr):
                corr = 0.0

            correlations.append(corr)

        except Exception as e:
            # Fallback for safety
            print(f"Warning: Error calculating Spearman for column {i}: {e}")
            correlations.append(0.0)

    return np.mean(correlations)
