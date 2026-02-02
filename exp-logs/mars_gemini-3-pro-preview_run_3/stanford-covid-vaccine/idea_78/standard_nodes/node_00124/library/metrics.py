import numpy as np
import torch
from library.config import Config


def compute_scored_mcrmse(predictions, targets):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) specifically
    for the scored subset of the data, as defined by the competition metric.

    This function:
    1. Slices the predictions and targets to the first 'seq_scored' positions (68).
    2. Filters the data to include only the 3 scored columns:
       'reactivity', 'deg_Mg_pH10', and 'deg_Mg_50C'.
    3. Computes the RMSE for each column independently.
    4. Returns the mean of these RMSE values.

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted values.
            Expected shape: (Batch_Size, Seq_Len_Pred, Num_Targets)
            e.g., (N, 107, 5)
        targets (np.ndarray or torch.Tensor): Ground truth values.
            Expected shape: (Batch_Size, Seq_Len_Tgt, Num_Targets)
            e.g., (N, 68, 5)

    Returns:
        float: The calculated MCRMSE score.
    """
    # 1. Convert to NumPy if inputs are PyTorch tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # 2. Slice Sequence Length
    # The metric is only evaluated on the first `seq_scored` positions (typically 68).
    # Predictions are usually length 107, targets are length 68.
    seq_scored = Config.SEQ_SCORED

    # Safety check: ensure we have enough data
    if predictions.shape[1] < seq_scored:
        raise ValueError(
            f"Predictions sequence length ({predictions.shape[1]}) is less than seq_scored ({seq_scored})"
        )

    # Slice predictions to (N, 68, 5)
    preds_sliced = predictions[:, :seq_scored, :]

    # Slice targets to (N, 68, 5) if they are longer (e.g. if padded)
    # If targets are already 68, this is a no-op or identity slice
    if targets.shape[1] >= seq_scored:
        targets_sliced = targets[:, :seq_scored, :]
    else:
        # If targets are shorter than seq_scored, we can't compute the metric properly
        raise ValueError(
            f"Targets sequence length ({targets.shape[1]}) is less than seq_scored ({seq_scored})"
        )

    # 3. Filter Scored Columns
    # Identify indices corresponding to the scored targets
    all_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_TARGETS

    # Find indices: e.g., reactivity is 0, deg_Mg_pH10 is 1, deg_Mg_50C is 3
    scored_indices = [i for i, col in enumerate(all_cols) if col in scored_cols]

    if not scored_indices:
        raise ValueError("No scored columns found in Config.")

    # Select only the relevant columns -> Shape (N, 68, 3)
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # 4. Compute RMSE per column
    # Calculate squared difference
    squared_diff = (preds_filtered - targets_filtered) ** 2

    # Mean Squared Error (MSE) per column: Average over Batch (0) and Sequence (1)
    # Result shape: (3,)
    mse_per_col = np.mean(squared_diff, axis=(0, 1))

    # Root Mean Squared Error (RMSE) per column
    rmse_per_col = np.sqrt(mse_per_col)

    # 5. Compute MCRMSE
    # Average the RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
