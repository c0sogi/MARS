import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored columns.

    Args:
        y_true (np.array): Ground truth values. Shape (N, seq_scored, 5) or (N*seq_scored, 5).
        y_pred (np.array): Predicted values. Shape (N, seq_scored, 5) or (N*seq_scored, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Identify indices of scored columns based on Config
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [Config.TARGET_COLS.index(col) for col in Config.SCORED_TARGETS]

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate RMSE for each scored column
    col_rmses = []
    for idx in scored_indices:
        # Extract specific column
        # Handle both flattened (N*L, C) and 3D (N, L, C) inputs
        if y_true.ndim == 3:
            y_t = y_true[:, :, idx].flatten()
            y_p = y_pred[:, :, idx].flatten()
        else:
            y_t = y_true[:, idx]
            y_p = y_pred[:, idx]

        # Calculate RMSE
        mse = np.mean((y_t - y_p) ** 2)
        rmse = np.sqrt(mse)
        col_rmses.append(rmse)

    # Return mean of RMSEs
    return np.mean(col_rmses)


def save_submission(preds, sample_ids, save_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the submission file.

    Args:
        preds (np.array): Predicted values. Shape (N_samples, seq_len, 5).
                          If seq_len is 68, it will be padded to 107.
        sample_ids (list): List of sample IDs corresponding to the predictions.
        save_path (str): Path to save the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Handle padding if necessary
    # The submission requires 107 positions, but model might output 68
    n_samples = preds.shape[0]
    seq_len = preds.shape[1]
    n_targets = preds.shape[2]

    final_preds = preds

    if seq_len == Config.SEQ_SCORED:
        # Pad with zeros to match SEQ_LENGTH (107)
        pad_len = Config.SEQ_LENGTH - seq_len
        padding = np.zeros((n_samples, pad_len, n_targets))
        final_preds = np.concatenate([preds, padding], axis=1)
    elif seq_len != Config.SEQ_LENGTH:
        # If it's neither 68 nor 107, log a warning but proceed
        print(
            f"Warning: Prediction sequence length is {seq_len}. Expected {Config.SEQ_SCORED} or {Config.SEQ_LENGTH}."
        )

    # Flatten predictions for dataframe construction
    # Shape becomes (N_samples * 107, 5)
    flat_preds = final_preds.reshape(-1, n_targets)

    # Generate ID_seqpos column
    # We need to repeat IDs and append sequence positions 0..106
    id_seqpos = []
    for sample_id in sample_ids:
        for i in range(Config.SEQ_LENGTH):
            id_seqpos.append(f"{sample_id}_{i}")

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", id_seqpos)

    # Save to CSV
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
