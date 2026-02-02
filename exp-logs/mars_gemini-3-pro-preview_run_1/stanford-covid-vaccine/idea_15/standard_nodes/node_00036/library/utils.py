import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each of the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    2. Taking the average of these column-wise RMSEs.

    This avoids the 'Mean of Sqrts' artifact by averaging RMSEs rather than MSEs.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values. Expected shape (N, seq_scored, 3).
        y_pred (np.ndarray or torch.Tensor): Predicted values. Expected shape (N, seq_scored, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes are consistent
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in MCRMSE calculation: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Mean Squared Error for each column
    # y_true/y_pred shape: (N_samples, seq_scored, N_columns)
    # We average over samples (axis 0) and sequence positions (axis 1)
    mse_per_column = np.mean((y_true - y_pred) ** 2, axis=(0, 1))

    # Calculate RMSE for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate the mean of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)


def build_submission_df(ids, preds, seq_len=107):
    """
    Formats predictions into the required submission DataFrame format.

    The output DataFrame has columns:
    ['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    The model predicts ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'].
    The unscored columns ['deg_pH10', 'deg_50C'] are filled with 0.0.

    Args:
        ids (list or np.ndarray): List of sample IDs.
        preds (np.ndarray or torch.Tensor): Predictions of shape (N_samples, seq_len, 3).
        seq_len (int): The length of the sequence (default 107).

    Returns:
        pd.DataFrame: The formatted submission DataFrame.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    submission_rows = []

    # Iterate over each sample
    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape: (seq_len, 3)

        # Iterate over each position in the sequence
        for pos in range(seq_len):
            # Extract predictions for the scored columns
            reactivity = sample_preds[pos, 0]
            deg_Mg_pH10 = sample_preds[pos, 1]
            deg_Mg_50C = sample_preds[pos, 2]

            # Fill unscored columns with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            # Construct the row ID
            id_seqpos = f"{sample_id}_{pos}"

            submission_rows.append(
                [id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]

    return pd.DataFrame(submission_rows, columns=columns)
