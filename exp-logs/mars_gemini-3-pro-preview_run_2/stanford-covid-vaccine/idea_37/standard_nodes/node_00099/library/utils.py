import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import TARGET_COLS, SCORED_COLS, SEQ_LENGTH, SCORING_LENGTH


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
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    for the scored columns only.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
                                             Shape: (N, seq_len, num_targets)
        y_pred (np.ndarray or torch.Tensor): Predicted values.
                                             Shape: (N, seq_len, num_targets)

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Identify indices of scored columns
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [i for i, col in enumerate(TARGET_COLS) if col in SCORED_COLS]

    # Filter arrays to scored columns only
    # y_true shape might be (Batch, Scored_Len, 5)
    y_true_scored = y_true[:, :, scored_indices]
    y_pred_scored = y_pred[:, :, scored_indices]

    # Calculate MSE per column (averaging over samples and sequence positions)
    # Result shape: (num_scored_cols,)
    mse_per_col = np.mean((y_true_scored - y_pred_scored) ** 2, axis=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Return the mean of the column-wise RMSEs
    return float(np.mean(rmse_per_col))


def format_submission(test_ids, preds, save_path):
    """
    Formats predictions into the competition submission format and saves to CSV.

    Args:
        test_ids (list): List of sample IDs from the test set.
        preds (np.ndarray or torch.Tensor): Model predictions of shape (N, 68, 5).
        save_path (str): Path to save the submission CSV.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    num_samples = len(test_ids)

    # Validate shapes
    # Expected preds shape: (240, 68, 5)
    assert (
        preds.shape[0] == num_samples
    ), f"Mismatch in samples: {preds.shape[0]} vs {num_samples}"
    assert preds.shape[2] == len(
        TARGET_COLS
    ), f"Mismatch in targets: {preds.shape[2]} vs {len(TARGET_COLS)}"

    # Prepare data container for the full length (107)
    # Initialize with zeros
    full_preds = np.zeros((num_samples, SEQ_LENGTH, len(TARGET_COLS)), dtype=np.float32)

    # Fill the scored positions (first 68)
    # If preds provided are longer than SCORING_LENGTH, truncate.
    # If shorter, this will raise an error (which is good).
    valid_len = min(preds.shape[1], SCORING_LENGTH)
    full_preds[:, :valid_len, :] = preds[:, :valid_len, :]

    # Reshape to (num_samples * SEQ_LENGTH, num_targets)
    # This flattens the data: ID1_pos0, ID1_pos1... ID1_pos106, ID2_pos0...
    flat_preds = full_preds.reshape(-1, len(TARGET_COLS))

    # Generate ID_seqpos keys
    # We repeat each ID 107 times
    ids_repeated = np.repeat(test_ids, SEQ_LENGTH)

    # We tile the sequence positions 0..106 for each sample
    seq_pos_tiled = np.tile(np.arange(SEQ_LENGTH), num_samples)

    # Combine to create "id_seqpos" strings
    # Vectorized string operation is faster than list comprehension
    id_seqpos = [f"{uid}_{pos}" for uid, pos in zip(ids_repeated, seq_pos_tiled)]

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=TARGET_COLS)
    submission_df.insert(0, "id_seqpos", id_seqpos)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save
    submission_df.to_csv(save_path, index=False)
