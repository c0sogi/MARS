import os
import random
import numpy as np
import torch
import torch.nn as nn
import hashlib
import json


def set_seed(seed=42):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Metric:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Where:
        Nt = number of target columns
        n = number of samples in the batch
        j = column index
        i = sample index
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, preds, targets):
        """
        Calculates the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predicted values. Shape (N, num_targets) or (N, seq_len, num_targets).
            targets (torch.Tensor): Ground truth values. Shape matches preds.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate MSE for each column (averaging over batch and sequence dimensions if present)
        # We flatten all dimensions except the last one (columns) to treat them as samples 'n'
        # preds.view(-1, preds.shape[-1]) ensures we have (Total_Samples, Num_Columns)

        # Note: If inputs are (Batch, Seq, Channels), we average over Batch and Seq.
        # dim=0 in the mean calculation below assumes flattened input or handles the reduction correctly.

        # Square error
        mse = torch.mean((preds - targets) ** 2, dim=0)

        # Root Mean Square error per column
        rmse = torch.sqrt(mse)

        # Mean of RMSEs across columns
        mcrmse = torch.mean(rmse)

        return mcrmse


def compute_mcrmse(preds, targets):
    """
    Computes MCRMSE using NumPy for validation/evaluation purposes.

    Args:
        preds (np.ndarray): Predicted values.
        targets (np.ndarray): Ground truth values.

    Returns:
        float: The MCRMSE score.
    """
    # Calculate MSE per column
    # axis=0 assumes preds/targets are (N, Columns) or flattened similarly.
    # If they are (N, Seq, Col), we flatten to (-1, Col) first.

    if preds.ndim == 3:
        preds = preds.reshape(-1, preds.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

    mse = np.mean((preds - targets) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return np.mean(rmse)


def get_hash(obj):
    """
    Generates an MD5 hash for a given object (e.g., config dictionary).
    Used for creating unique cache filenames.

    Args:
        obj (any): Object to hash. Should be serializable to string or JSON.

    Returns:
        str: MD5 hash string.
    """
    if isinstance(obj, dict):
        # Sort keys to ensure consistent ordering for dictionaries
        obj_str = json.dumps(obj, sort_keys=True)
    else:
        obj_str = str(obj)

    return hashlib.md5(obj_str.encode("utf-8")).hexdigest()
