import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Configures CUDA to be deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricTracker:
    """
    Accumulates error metrics globally across batches to compute MCRMSE correctly.
    This avoids the bias introduced by averaging RMSEs of mini-batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Accumulators for Sum of Squared Errors (SSE) and Counts per column
        # Shape: (5,) corresponding to the 5 output channels
        self.sse = np.zeros(5, dtype=np.float64)
        self.count = np.zeros(5, dtype=np.float64)

        # Indices of columns used for the competition metric:
        # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
        self.scored_cols = [0, 1, 3]

    def update(self, preds, targets):
        """
        Updates the running statistics with a new batch of data.

        Args:
            preds: Predictions tensor or array (Batch, Length, 5)
            targets: Ground truth tensor or array (Batch, Length, 5)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Ensure shapes match
        if preds.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
            )

        # Compute squared errors element-wise
        squared_errors = (preds - targets) ** 2

        # Sum errors over batch and sequence length dimensions (axes 0 and 1)
        # Result shape: (5,)
        batch_sse = np.sum(squared_errors, axis=(0, 1))

        # Count number of elements contributing to the sum (Batch * Length)
        # Note: Assumes inputs are already sliced/masked to the valid scored region
        batch_count = preds.shape[0] * preds.shape[1]

        self.sse += batch_sse
        self.count += batch_count

    def result(self):
        """
        Computes the global MCRMSE over the accumulated data.

        Returns:
            float: The mean column-wise root mean squared error for the scored columns.
        """
        # Avoid division by zero
        safe_count = np.maximum(self.count, 1e-8)

        # RMSE per column = sqrt(Total SSE / Total Count)
        rmse_per_col = np.sqrt(self.sse / safe_count)

        # Filter for scored columns only (reactivity, deg_Mg_pH10, deg_Mg_50C)
        scored_rmse = rmse_per_col[self.scored_cols]

        # MCRMSE is the mean of the RMSEs of the scored columns
        return np.mean(scored_rmse)


def format_submission(preds, ids, save_path=Config.SUBMISSION_PATH):
    """
    Formats raw predictions into the competition submission CSV format.

    Args:
        preds: List of arrays or np.array of shape (N_samples, Seq_Len, 5)
        ids: List of sample IDs (N_samples)
        save_path: Output path for the CSV file.
    """
    if isinstance(preds, list):
        preds = np.array(preds)

    # Validation of shapes
    if len(ids) != preds.shape[0]:
        raise ValueError("Number of IDs does not match number of prediction samples.")

    num_samples = preds.shape[0]
    seq_len = preds.shape[1]  # Should be 107
    num_targets = preds.shape[2]  # Should be 5

    # Target column names in order
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    if num_targets != len(cols):
        raise ValueError(f"Expected {len(cols)} target columns, got {num_targets}")

    # Create the 'id_seqpos' column
    # Repeat each ID 'seq_len' times: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(ids, seq_len)

    # Tile the sequence positions (0..106) 'num_samples' times: [0, 1, ..., 0, 1, ...]
    seq_positions = np.tile(np.arange(seq_len), num_samples)

    # Combine into strings "id_seqpos"
    id_seqpos_col = [f"{i}_{p}" for i, p in zip(ids_repeated, seq_positions)]

    # Flatten predictions to (N_samples * Seq_Len, 5)
    preds_flat = preds.reshape(-1, num_targets)

    # Create DataFrame
    df = pd.DataFrame(preds_flat, columns=cols)
    df.insert(0, "id_seqpos", id_seqpos_col)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
