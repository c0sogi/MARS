import os
import random
import ast
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_list_column(x: str) -> np.ndarray:
    """
    Parses a string representation of a list (e.g., '[0.1, 0.2]') into a NumPy array.
    Used for loading target columns from the metadata CSVs.

    Args:
        x (str): Stringified list.

    Returns:
        np.ndarray: Array of float32 values. Returns an empty array if parsing fails.
    """
    try:
        # ast.literal_eval is safer than eval
        val = ast.literal_eval(x)
        return np.array(val, dtype=np.float32)
    except (ValueError, SyntaxError, TypeError):
        return np.array([], dtype=np.float32)


def get_score(
    y_true: np.ndarray, y_pred: np.ndarray, scored_indices: list = None
) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    MCRMSE = Average across columns of (RMSE for that column).

    Args:
        y_true (np.ndarray): Ground truth array of shape (N, L, C) or (N, C).
        y_pred (np.ndarray): Predicted array of shape (N, L, C) or (N, C).
        scored_indices (list, optional): List of column indices to include in the score.
                                         If None, all columns are used.

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Select specific columns if requested
    if scored_indices is not None:
        y_true = y_true[..., scored_indices]
        y_pred = y_pred[..., scored_indices]

    # Calculate MSE per element
    squared_diff = (y_true - y_pred) ** 2

    # Calculate RMSE per column
    # If shape is (N, L, C), we average over N and L (axis 0 and 1)
    # If shape is (N, C), we average over N (axis 0)
    if y_true.ndim == 3:
        # Average over samples and sequence length
        mse_per_col = np.nanmean(squared_diff, axis=(0, 1))
    else:
        # Average over samples
        mse_per_col = np.nanmean(squared_diff, axis=0)

    rmse_per_col = np.sqrt(mse_per_col)

    # Mean of RMSEs across columns
    mcrmse = np.nanmean(rmse_per_col)

    return float(mcrmse)


def format_submission(
    test_ids: list, predictions: np.ndarray, save_path: str = Config.SUBMISSION_PATH
):
    """
    Formats predictions into the competition submission CSV format.

    The format requires one row per sequence position:
    id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    Args:
        test_ids (list): List of sample IDs (strings).
        predictions (np.ndarray): Array of shape (num_samples, seq_len, 5).
                                  Must cover the full sequence length (107).
        save_path (str): Path to save the CSV file.
    """
    num_samples, seq_len, num_targets = predictions.shape

    # Ensure we have the correct number of targets
    assert num_targets == 5, f"Expected 5 target columns, got {num_targets}"
    assert (
        len(test_ids) == num_samples
    ), "Mismatch between number of IDs and prediction samples"

    # 1. Create the 'id_seqpos' column
    # Repeat IDs seq_len times: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(test_ids, seq_len)

    # Tile sequence positions: [0, 1, ..., 106, 0, 1, ..., 106]
    seq_pos = np.tile(np.arange(seq_len), num_samples)

    # Combine into strings
    id_seqpos = [f"{i}_{p}" for i, p in zip(ids_repeated, seq_pos)]

    # 2. Flatten predictions
    # Reshape from (N, L, 5) to (N*L, 5)
    preds_flat = predictions.reshape(-1, num_targets)

    # 3. Create DataFrame
    df_sub = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)
    df_sub.insert(0, "id_seqpos", id_seqpos)

    # 4. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}. Shape: {df_sub.shape}")
