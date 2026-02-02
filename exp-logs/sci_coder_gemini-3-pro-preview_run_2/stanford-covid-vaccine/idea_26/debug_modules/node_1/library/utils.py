import os
import random
import ast
import numpy as np
import torch
import torch.nn as nn
from library import config


def set_seed(seed=42):
    """
    Sets the seed for random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_list_column(x):
    """
    Parses a stringified list (e.g., '[0.1, 0.2]') into a numpy array.
    Returns an empty array if parsing fails or input is invalid.
    """
    try:
        if isinstance(x, str):
            # Evaluate the string as a python literal (list)
            val_list = ast.literal_eval(x)
            return np.array(val_list, dtype=np.float32)
        elif isinstance(x, (list, tuple, np.ndarray)):
            return np.array(x, dtype=np.float32)
        return np.array([], dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates the loss only on the scored columns defined in config.SCORED_INDICES.
    """

    def __init__(self):
        super().__init__()
        self.scored_indices = config.SCORED_INDICES

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted tensor of shape (Batch, SeqLen, Targets)
            target: Ground truth tensor of shape (Batch, SeqLen, Targets)
            mask: Boolean or Float mask of shape (Batch, SeqLen) indicating valid positions.
                  Should be 1.0 for valid positions (first 68) and 0.0 otherwise.
        """
        rmse_list = []

        # Iterate over only the columns that are scored
        for idx in self.scored_indices:
            p = pred[:, :, idx]
            t = target[:, :, idx]

            diff_sq = (p - t) ** 2

            if mask is not None:
                # Apply mask to squared differences
                diff_sq = diff_sq * mask

                # Compute mean over valid elements
                # Sum of squared errors / Number of valid elements
                # We add a small epsilon to count to avoid division by zero
                count = mask.sum() + 1e-8
                mse = diff_sq.sum() / count
            else:
                mse = diff_sq.mean()

            rmse_list.append(torch.sqrt(mse))

        # MCRMSE is the mean of the RMSEs of the columns
        return torch.mean(torch.stack(rmse_list))
