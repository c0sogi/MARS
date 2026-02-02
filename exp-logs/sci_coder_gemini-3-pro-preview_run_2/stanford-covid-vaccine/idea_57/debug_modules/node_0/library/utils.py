import os
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_list_column(x):
    """
    Parses a stringified list (e.g., '[0.1, 0.2]') into a numpy array.
    Returns an empty array if parsing fails or input is invalid.
    """
    try:
        if isinstance(x, str):
            # ast.literal_eval is safer than eval
            return np.array(ast.literal_eval(x), dtype=np.float32)
        return np.array(x, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates loss strictly on scored columns and valid sequence positions (0-67).
    """

    def __init__(self):
        super().__init__()
        # Identify indices of scored columns within the full target list
        # Config.TARGET_COLS contains all 5 columns
        # Config.SCORED_COLS contains the 3 scored columns
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        # Register as buffer to handle device movement automatically, but not a model parameter
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )
        self.seq_scored = Config.SCORED_LENGTH

    def forward(self, preds, targets):
        """
        Args:
            preds: (Batch, Seq_Len, 5) - Predictions for all columns and positions
            targets: (Batch, Seq_Len, 5) - Ground truth
        Returns:
            mcrmse: Scalar loss value
        """
        # 1. Select only the scored sequence positions (usually 0 to 67)
        preds_sliced = preds[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # 2. Select only the scored columns (Reactivity, Deg_Mg_pH10, Deg_Mg_50C)
        preds_scored = torch.index_select(preds_sliced, 2, self.scored_indices_tensor)
        targets_scored = torch.index_select(
            targets_sliced, 2, self.scored_indices_tensor
        )

        # 3. Compute MSE per column (averaging over Batch and Sequence dimensions)
        mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Average RMSE across the scored columns to get MCRMSE
        mcrmse = torch.mean(rmse)

        return mcrmse


def calculate_global_mcrmse(all_preds, all_targets):
    """
    Calculates MCRMSE over the entire dataset.
    Accumulates errors globally before rooting to avoid batch-averaging bias.

    Args:
        all_preds: List of arrays or single array (N, Seq, 5)
        all_targets: List of arrays or single array (N, Seq, 5)

    Returns:
        float: The global MCRMSE score.
    """
    # Concatenate if inputs are lists of batch outputs
    if isinstance(all_preds, list):
        all_preds = np.concatenate(all_preds, axis=0)
    if isinstance(all_targets, list):
        all_targets = np.concatenate(all_targets, axis=0)

    # Indices for scored columns
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Slice sequence length to scored region
    seq_len = Config.SCORED_LENGTH
    preds_sliced = all_preds[:, :seq_len, :]
    targets_sliced = all_targets[:, :seq_len, :]

    # Select scored columns
    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = targets_sliced[:, :, scored_indices]

    # Calculate squared errors: (N, Seq, Cols)
    squared_errors = (preds_scored - targets_scored) ** 2

    # Mean over N and Seq dimensions for each column
    # This effectively computes Sum(SSE) / Total_Count for each column
    mse_per_col = np.mean(squared_errors, axis=(0, 1))

    # RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Mean of RMSEs across columns
    global_mcrmse = np.mean(rmse_per_col)

    return float(global_mcrmse)


def format_submission(test_ids, preds, save_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the competition submission format and saves to CSV.

    Args:
        test_ids: List or array of sample IDs (N,)
        preds: Numpy array of predictions (N, 107, 5)
        save_path: Path to save the CSV
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_samples = len(test_ids)
    seq_len = preds.shape[1]  # Should be 107

    # Generate 'id_seqpos' column
    flat_ids = []
    for sample_id in test_ids:
        for i in range(seq_len):
            flat_ids.append(f"{sample_id}_{i}")

    # Flatten predictions from (N, 107, 5) to (N*107, 5)
    preds_flat = preds.reshape(-1, 5)

    # Create DataFrame
    submission_df = pd.DataFrame({"id_seqpos": flat_ids})

    # Add prediction columns in the correct order
    for i, col in enumerate(Config.TARGET_COLS):
        submission_df[col] = preds_flat[:, i]

    # Save to CSV
    submission_df.to_csv(save_path, index=False)
