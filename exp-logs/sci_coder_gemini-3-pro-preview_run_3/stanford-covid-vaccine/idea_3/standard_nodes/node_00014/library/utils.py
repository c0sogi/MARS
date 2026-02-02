import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can be slower, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device to use for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Formula:
    MCRMSE = (1/Nt) * Sum_{j=1}^{Nt} sqrt( (1/n) * Sum_{i=1}^{n} (y_{ij} - y_hat_{ij})^2 )

    Where:
    - Nt is the number of target columns (5).
    - n is the total number of elements per column (Batch Size * Sequence Length).
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets, col_indices=None):
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values. Shape (Batch, Seq_Len, Targets) or (Batch, Targets).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len, Targets) or (Batch, Targets).
            col_indices (list, optional): List of indices to include in the loss calculation.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if col_indices is not None:
            inputs = inputs[..., col_indices]
            targets = targets[..., col_indices]

        # Calculate Squared Error: (y - y_hat)^2
        squared_diff = (inputs - targets) ** 2

        # Calculate Mean Squared Error per column
        # We average over all dimensions except the last one (which represents the targets)
        # dims_to_reduce will be [0, 1] for 3D input (Batch, Seq, Targets)
        # or [0] for 2D input (Batch, Targets)
        dims_to_reduce = list(range(len(inputs.shape) - 1))

        mse = torch.mean(squared_diff, dim=dims_to_reduce)

        # Calculate RMSE per column
        rmse = torch.sqrt(mse)

        # Calculate Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse)

        return mcrmse


def format_submission(preds, sample_ids):
    """
    Formats the predictions into the required submission DataFrame format.

    Args:
        preds (np.ndarray): Predictions of shape (N_samples, Seq_Len, 5).
                            Seq_Len should be 107.
                            Columns order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C.
        sample_ids (list or np.ndarray): List of sample_id strings corresponding to the predictions.

    Returns:
        pd.DataFrame: Formatted dataframe ready for CSV export.
    """
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    n_samples = len(sample_ids)
    seq_len = preds.shape[1]
    n_targets = preds.shape[2]

    # Ensure shapes match expectations
    assert seq_len == 107, f"Expected sequence length 107, got {seq_len}"
    assert n_targets == 5, f"Expected 5 targets, got {n_targets}"
    assert (
        n_samples == preds.shape[0]
    ), f"Mismatch between IDs ({n_samples}) and predictions ({preds.shape[0]})"

    # Flatten predictions to (N_samples * Seq_Len, 5)
    preds_flat = preds.reshape(-1, n_targets)

    # Generate id_seqpos identifiers efficiently
    # Repeat IDs: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(sample_ids, seq_len)

    # Tile seqpos: [0, 1, ..., 106, 0, 1, ..., 106]
    seqpos_tiled = np.tile(np.arange(seq_len), n_samples)

    # Combine into strings: "id_seqpos"
    # Using list comprehension as it is reasonably fast for ~25k items
    id_seqpos = [f"{i}_{s}" for i, s in zip(ids_repeated, seqpos_tiled)]

    # Create DataFrame
    submission_df = pd.DataFrame(preds_flat, columns=target_cols)
    submission_df.insert(0, "id_seqpos", id_seqpos)

    return submission_df
