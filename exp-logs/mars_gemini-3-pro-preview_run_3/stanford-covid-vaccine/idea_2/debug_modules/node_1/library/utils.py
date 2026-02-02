import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_pred, y_true, num_scored=Config.SEQ_SCORED):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the mean of the RMSEs for each target column,
    evaluated only on the first `num_scored` positions of the sequence.

    Args:
        y_pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Targets).
        y_true (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Targets).
        num_scored (int): Number of positions along the sequence to include in the loss.
                          Defaults to Config.SEQ_SCORED (68).

    Returns:
        torch.Tensor: Scalar MCRMSE loss.
    """
    # Slice to consider only the scored positions along the sequence dimension (dim 1)
    # y_pred and y_true shapes: (Batch, Sequence, Targets)
    y_pred_scored = y_pred[:, :num_scored, :]
    y_true_scored = y_true[:, :num_scored, :]

    # Calculate MSE per column (averaging over Batch (dim 0) and Sequence (dim 1) dimensions)
    mse = torch.mean((y_true_scored - y_pred_scored) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs across columns (dim 0 of the rmse vector)
    mcrmse = torch.mean(rmse)

    return mcrmse


def format_submission(ids, preds, save_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the required CSV format and saves to disk.

    Args:
        ids (list): List of sample IDs (strings).
        preds (np.ndarray or torch.Tensor): Predictions of shape (N_samples, Seq_Len, 5).
                                            Seq_Len must be 107.
        save_path (str): Path to save the CSV file. Defaults to Config.SUBMISSION_PATH.

    Returns:
        pd.DataFrame: The formatted submission DataFrame.
    """
    # Convert tensor to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    # Validation of shapes
    if preds.shape[1] != Config.SEQ_LENGTH:
        raise ValueError(
            f"Prediction sequence length must be {Config.SEQ_LENGTH}, got {preds.shape[1]}"
        )
    if preds.shape[2] != Config.NUM_TARGETS:
        raise ValueError(
            f"Prediction must have {Config.NUM_TARGETS} target columns, got {preds.shape[2]}"
        )
    if len(ids) != preds.shape[0]:
        raise ValueError(
            f"Number of IDs ({len(ids)}) does not match number of predictions ({preds.shape[0]})"
        )

    # Prepare lists for DataFrame construction
    id_seqpos_list = []
    # Initialize lists for each target column
    target_data = {col: [] for col in Config.TARGET_COLS}

    # Iterate through each sample
    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape (107, 5)

        # Iterate through each sequence position (0 to 106)
        for seq_idx in range(Config.SEQ_LENGTH):
            # Create unique id_seqpos identifier
            id_seqpos = f"{sample_id}_{seq_idx}"
            id_seqpos_list.append(id_seqpos)

            # Append predictions for each target column
            for col_idx, col_name in enumerate(Config.TARGET_COLS):
                target_data[col_name].append(sample_preds[seq_idx, col_idx])

    # Construct DataFrame
    submission_df = pd.DataFrame({"id_seqpos": id_seqpos_list})
    for col_name in Config.TARGET_COLS:
        submission_df[col_name] = target_data[col_name]

    # Save to CSV if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission_df.to_csv(save_path, index=False)

    return submission_df
