import torch
import numpy as np
from library.config import Config, seed_everything


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility by wrapping the library function.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    seed_everything(seed)


def scored_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the
    competition-specific scored targets and sequence positions.

    This function handles:
    1. Slicing sequences to the scored length (Config.SEQ_SCORED = 68).
    2. Filtering columns to the scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Computing RMSE per column and averaging them.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
            Expected shape: (Batch, 68, 5) or (Batch, 107, 5).
        y_pred (torch.Tensor or np.ndarray): Predicted values.
            Expected shape: (Batch, 107, 5) or (Batch, 68, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert numpy arrays to torch tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Detach from graph if necessary and move to CPU for metric calculation
    if y_pred.requires_grad:
        y_pred = y_pred.detach()

    y_true = y_true.cpu().float()
    y_pred = y_pred.cpu().float()

    # 1. Slice Sequence Dimension
    # The competition only scores the first 68 bases
    seq_len_scored = Config.SEQ_SCORED

    # Ensure prediction has enough length
    if y_pred.shape[1] < seq_len_scored:
        raise ValueError(
            f"Prediction sequence length {y_pred.shape[1]} is less than scored length {seq_len_scored}"
        )

    # Slice predictions to (Batch, 68, 5)
    y_pred_sliced = y_pred[:, :seq_len_scored, :]

    # Slice targets if necessary (Batch, 68+, 5) -> (Batch, 68, 5)
    if y_true.shape[1] >= seq_len_scored:
        y_true_sliced = y_true[:, :seq_len_scored, :]
    else:
        raise ValueError(
            f"Target sequence length {y_true.shape[1]} is less than scored length {seq_len_scored}"
        )

    # 2. Filter Scored Columns
    # Identify indices of the columns that count towards the score
    all_targets = Config.TARGET_COLS
    scored_targets = Config.SCORED_TARGETS

    # Map target names to indices (e.g., [0, 1, 3])
    target_indices = [all_targets.index(t) for t in scored_targets]

    # Select only the scored columns
    y_pred_filtered = y_pred_sliced[:, :, target_indices]
    y_true_filtered = y_true_sliced[:, :, target_indices]

    # 3. Compute Metric
    # MSE per column: Average over Batch (dim 0) and Sequence (dim 1)
    mse_per_col = torch.mean((y_true_filtered - y_pred_filtered) ** 2, dim=(0, 1))

    # RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # MCRMSE: Mean of the column RMSEs
    mcrmse = torch.mean(rmse_per_col)

    return mcrmse.item()
