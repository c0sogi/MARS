import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config, set_seed


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Metric Formula:
        MCRMSE = (1/Nt) * sum_{j=1}^{Nt} RMSE_j
        where RMSE_j = sqrt( (1/N) * sum_{i=1}^{N} (y_{ij} - y_hat_{ij})^2 )

    Nt = Number of target columns (5)
    N = Total number of scored positions (Batch Size * Scored Sequence Length)
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Forward pass for MCRMSE Loss.

        Args:
            inputs (torch.Tensor): Predicted values of shape (batch_size, seq_len, num_targets).
            targets (torch.Tensor): Ground truth values of shape (batch_size, seq_len, num_targets).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Ensure inputs and targets are float tensors
        inputs = inputs.float()
        targets = targets.float()

        # Calculate Squared Error: (y - y_hat)^2
        squared_diff = (inputs - targets) ** 2

        # Calculate MSE per column (averaging over batch and sequence length)
        # dim=0 is batch, dim=1 is sequence length
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate Mean of RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_column)

        return loss


def process_submission(preds, test_ids, save_path=Config.SUBMISSION_PATH):
    """
    Formats prediction array into the competition submission CSV format and saves it.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions array of shape (num_samples, seq_len, num_targets).
                                            Expected seq_len is 107.
        test_ids (list or np.ndarray): List of sample IDs corresponding to the first dimension of preds.
        save_path (str): File path to save the submission CSV.
    """
    # Ensure preds is numpy array
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    num_samples, seq_len, num_targets = preds.shape

    # Validation
    if num_targets != len(Config.TARGET_COLS):
        raise ValueError(
            f"Number of prediction targets ({num_targets}) does not match Config ({len(Config.TARGET_COLS)})"
        )

    if seq_len != Config.SEQ_LEN:
        # Warning only, as logic handles dynamic length, but 107 is expected for this task
        print(
            f"Warning: Prediction sequence length is {seq_len}, expected {Config.SEQ_LEN}."
        )

    # Flatten predictions to 2D array: (num_samples * seq_len, num_targets)
    preds_flat = preds.reshape(-1, num_targets)

    # Generate id_seqpos column
    # Repeat IDs: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(test_ids, seq_len)

    # Tile sequence positions: [0, 1, ..., 106, 0, 1, ..., 106, ...]
    seq_pos_tiled = np.tile(np.arange(seq_len), num_samples)

    # Combine to create identifiers: id_seqpos
    id_seqpos_col = [f"{uid}_{pos}" for uid, pos in zip(ids_repeated, seq_pos_tiled)]

    # Create DataFrame
    submission_df = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", id_seqpos_col)

    # Save to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
