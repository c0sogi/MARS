import torch
import numpy as np
import os
import random
from library.config import Config, set_seed


def get_device():
    """
    Returns the PyTorch device configured in Config.

    Returns:
        torch.device: The device (cuda or cpu).
    """
    return torch.device(Config.DEVICE)


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the library.config.set_seed function.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    set_seed(seed)


class MCRMSE:
    """
    Mean Columnwise Root Mean Squared Error Metric.

    Calculates the RMSE for each target column separately, then takes the mean
    of these RMSE values. This metric is specific to the competition scoring.

    Formula:
    MCRMSE = (1/N_t) * Sum_{j=1}^{N_t} sqrt( (1/n) * Sum_{i=1}^{n} (y_{ij} - y_hat_{ij})^2 )

    Where:
        N_t: Number of target columns (3 for this task).
        n: Total number of scored positions across the batch.
    """

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        """
        Computes the MCRMSE score.

        Args:
            y_true (torch.Tensor): Ground truth tensor of shape (Batch, Seq_Len, Num_Targets).
            y_pred (torch.Tensor): Prediction tensor of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: A scalar tensor containing the MCRMSE score.
        """
        # Ensure inputs are floating point tensors
        y_true = y_true.float()
        y_pred = y_pred.float()

        # Calculate squared errors: (y - y_hat)^2
        squared_errors = (y_true - y_pred) ** 2

        # Calculate Mean Squared Error (MSE) for each column j.
        # We flatten the batch and sequence dimensions (dim 0 and 1) into a single dimension 'n'.
        # view(-1, num_targets) collapses (Batch, Seq_Len) -> (Batch * Seq_Len)
        mse_per_column = torch.mean(
            squared_errors.view(-1, squared_errors.shape[-1]), dim=0
        )

        # Calculate Root Mean Squared Error (RMSE) for each column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate the mean across the columns (targets)
        mcrmse_score = torch.mean(rmse_per_column)

        return mcrmse_score


def tokenize_sequence(sequence):
    """
    Converts an RNA nucleotide sequence string into a numpy array of integer tokens.
    Uses Config.TOKEN_VOCAB for mapping.

    Args:
        sequence (str): The RNA sequence (e.g., "GGAAUC...").

    Returns:
        np.array: Array of integers representing the sequence.
    """
    # Use .get() to handle potential unexpected characters safely, though data should be clean.
    # Defaulting to -1 which would cause an error downstream if not handled,
    # ensuring we catch data issues early.
    return np.array([Config.TOKEN_VOCAB[char] for char in sequence], dtype=np.int64)


def tokenize_loop_type(loop_type_str):
    """
    Converts a predicted loop type string into a numpy array of integer tokens.
    Uses Config.LOOP_VOCAB for mapping.

    Args:
        loop_type_str (str): The loop type sequence (e.g., "EEEEESSS...").

    Returns:
        np.array: Array of integers representing the loop types.
    """
    return np.array([Config.LOOP_VOCAB[char] for char in loop_type_str], dtype=np.int64)


def get_scored_columns():
    """
    Returns the list of column names that are used for scoring.

    Returns:
        list: List of strings ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"].
    """
    return Config.TARGET_COLS
