import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the
    scored columns and positions.

    The metric is calculated only on the first 68 positions (seq_scored) and
    specific columns: reactivity, deg_Mg_pH10, and deg_Mg_50C.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values. Shape (B, 68, 5).
        y_pred (torch.Tensor or np.ndarray): Predicted values. Shape (B, 107, 5) or (B, 68, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are torch tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Move to CPU to ensure device compatibility
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    # Determine scored sequence length (usually 68)
    seq_scored = y_true.shape[1]

    # Slice predictions to match the scored length
    # y_pred might be length 107, we only care about the first seq_scored positions
    y_pred_sliced = y_pred[:, :seq_scored, :]

    # Identify indices of scored columns based on Config.TARGET_COLS
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Scored: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_col_indices = [0, 1, 3]

    # Filter for scored columns
    y_true_filtered = y_true[:, :, scored_col_indices]
    y_pred_filtered = y_pred_sliced[:, :, scored_col_indices]

    # Calculate MSE for each column (averaging over batch and sequence length)
    # Result shape: (3,)
    column_mse = torch.mean((y_true_filtered - y_pred_filtered) ** 2, dim=(0, 1))

    # Calculate RMSE for each column
    column_rmse = torch.sqrt(column_mse)

    # Calculate Mean of RMSEs
    mcrmse = torch.mean(column_rmse)

    return mcrmse.item()


def format_submission(test_ids, preds, save_path):
    """
    Formats the predictions into the submission CSV format and saves it.

    Args:
        test_ids (list): List of sample IDs from the test set.
        preds (np.ndarray): Prediction array of shape (N_samples, 107, 5).
        save_path (str): Path to save the CSV file.
    """
    # Validate shapes
    if len(test_ids) != preds.shape[0]:
        raise ValueError(
            f"Mismatch between ID count ({len(test_ids)}) and prediction count ({preds.shape[0]})"
        )

    N_samples, Seq_Len, N_targets = preds.shape

    # Flatten predictions to 2D array: (N_samples * Seq_Len, N_targets)
    preds_flat = preds.reshape(-1, N_targets)

    # Generate 'id_seqpos' identifiers
    # Logic: for each id, create id_0, id_1, ..., id_106
    id_seqpos_list = [f"{sid}_{pos}" for sid in test_ids for pos in range(Seq_Len)]

    # Create DataFrame
    df = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)
    df.insert(0, "id_seqpos", id_seqpos_list)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
