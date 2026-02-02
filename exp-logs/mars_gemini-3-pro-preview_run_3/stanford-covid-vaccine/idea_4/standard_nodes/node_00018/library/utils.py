import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across all libraries.
    Delegates to the Config class method.
    """
    Config.set_seed(seed)


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Metric:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Where:
    - j iterates over the target columns (Nt=5).
    - i iterates over the samples and sequence positions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Forward pass for MCRMSE Loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels) or (Batch, Channels).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Channels) or (Batch, Channels).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Determine dimensions to reduce over (Batch and Sequence)
        # If input is (Batch, Seq, Channels), reduce over dim 0 and 1.
        # If input is (Batch, Channels), reduce over dim 0.
        if preds.dim() == 3:
            reduce_dims = (0, 1)
        else:
            reduce_dims = (0,)

        # 1. Calculate Mean Squared Error for each column (channel)
        mse = torch.mean((preds - targets) ** 2, dim=reduce_dims)

        # 2. Calculate Root Mean Squared Error for each column
        rmse = torch.sqrt(mse)

        # 3. Calculate Mean of RMSEs across columns
        mcrmse = torch.mean(rmse)

        return mcrmse


def compute_mcrmse(preds, targets):
    """
    Computes the MCRMSE score using Numpy for validation/evaluation.

    Args:
        preds (np.ndarray or torch.Tensor): Model predictions.
        targets (np.ndarray or torch.Tensor): Ground truth values.

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Determine axes to reduce over
    if preds.ndim == 3:
        reduce_axes = (0, 1)
    else:
        reduce_axes = (0,)

    # Calculate MSE per column
    mse = np.mean((preds - targets) ** 2, axis=reduce_axes)

    # Calculate RMSE per column
    rmse = np.sqrt(mse)

    # Average RMSEs
    score = np.mean(rmse)

    return float(score)


def create_submission(ids, preds, save_path=Config.FINAL_SUBMISSION):
    """
    Generates and saves the submission CSV file in the required format.

    Format:
    id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C

    Args:
        ids (list or np.ndarray): List of sample IDs (length N).
        preds (np.ndarray): Predictions tensor of shape (N, 107, 5).
        save_path (str): Path to save the CSV file.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # Ensure inputs are valid
    if len(ids) != preds.shape[0]:
        raise ValueError(
            f"Mismatch between IDs count ({len(ids)}) and predictions samples ({preds.shape[0]})."
        )

    if preds.shape[1] != Config.SEQ_LENGTH:
        raise ValueError(
            f"Predictions sequence length ({preds.shape[1]}) does not match Config ({Config.SEQ_LENGTH})."
        )

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    n_samples = len(ids)
    seq_len = preds.shape[1]

    # 1. Generate 'id_seqpos' column
    # Repeat IDs for each sequence position: [id1, id1... id2, id2...]
    sample_ids_repeated = np.repeat(ids, seq_len)

    # Tile sequence positions: [0, 1... 106, 0, 1... 106]
    seq_pos_tiled = np.tile(np.arange(seq_len), n_samples)

    # Combine into strings
    id_seqpos = [f"{s}_{p}" for s, p in zip(sample_ids_repeated, seq_pos_tiled)]

    # 2. Flatten predictions to (N * 107, 5)
    preds_flat = preds.reshape(-1, 5)

    # 3. Create DataFrame
    df = pd.DataFrame(preds_flat, columns=target_cols)
    df.insert(0, "id_seqpos", id_seqpos)

    # 4. Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {df.shape}")

    return df
