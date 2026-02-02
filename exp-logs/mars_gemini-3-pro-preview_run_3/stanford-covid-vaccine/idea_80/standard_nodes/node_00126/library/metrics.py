import torch
from library.config import Config
from library.utils import mcrmse


def calculate_competition_metric(
    preds: torch.Tensor, targets: torch.Tensor, config: Config
) -> torch.Tensor:
    """
    Calculates the official competition metric (MCRMSE) on the scored positions and columns.

    This function performs the necessary slicing and filtering required to match the
    Kaggle leaderboard evaluation:
    1. Slices the sequence dimension to the first `config.pred_len` (68) positions.
    2. Filters the target columns to only include `config.scored_cols`
       ('reactivity', 'deg_Mg_pH10', 'deg_Mg_50C').
    3. Flattens the data to treat every position in the batch as an independent sample.
    4. Computes the MCRMSE.

    Args:
        preds (torch.Tensor): Predicted values.
                              Shape: (Batch_Size, Seq_Len, Num_Targets)
        targets (torch.Tensor): Ground truth values.
                                Shape: (Batch_Size, Seq_Len, Num_Targets)
        config (Config): Configuration object containing pred_len, target_cols,
                         and scored_cols.

    Returns:
        torch.Tensor: The scalar MCRMSE score calculated over the specific subset
                      of data used for leaderboard scoring.
    """
    # 1. Slice to scored sequence length (first 68 bases)
    # The competition only scores the first 68 bases, even though sequences are 107 long.
    # Shape becomes: (Batch, 68, 5)
    preds_sliced = preds[:, : config.pred_len, :]
    targets_sliced = targets[:, : config.pred_len, :]

    # 2. Identify indices of scored columns
    # We train on 5 targets, but only 3 are scored.
    scored_indices = [
        i for i, col in enumerate(config.target_cols) if col in config.scored_cols
    ]

    if not scored_indices:
        raise ValueError(
            "No scored columns found in config.target_cols matching config.scored_cols."
        )

    # Select specific columns
    # Shape becomes: (Batch, 68, 3)
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # 3. Flatten for MCRMSE calculation
    # The MCRMSE metric is defined as the mean of RMSEs per column.
    # The RMSE is calculated over all samples (n) where n = Batch * Seq_Len.
    # We reshape to (-1, Num_Scored_Targets) so that dim=0 represents 'n'.
    num_scored = len(config.scored_cols)

    preds_flat = preds_filtered.reshape(-1, num_scored)
    targets_flat = targets_filtered.reshape(-1, num_scored)

    # 4. Compute Metric
    return mcrmse(targets_flat, preds_flat)
