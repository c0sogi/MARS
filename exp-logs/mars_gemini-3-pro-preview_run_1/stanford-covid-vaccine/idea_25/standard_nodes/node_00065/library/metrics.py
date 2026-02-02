import numpy as np
import torch


def compute_mcrmse(preds, targets, mask=None):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing RMSE separately for each target column (reactivity, deg_Mg_pH10, etc.).
    2. Averaging the RMSE values across all columns.

    This approach avoids the 'Mean of Sqrts' vs 'Sqrt of Means' ambiguity by strictly
    following the competition metric definition: average of column-wise RMSEs.

    Args:
        preds (Union[np.ndarray, torch.Tensor]): Predicted values.
            Shape: (Batch, Seq_Len, Channels)
        targets (Union[np.ndarray, torch.Tensor]): Ground truth values.
            Shape: (Batch, Seq_Len, Channels)
        mask (Union[np.ndarray, torch.Tensor], optional): Boolean mask indicating valid positions.
            Shape: (Batch, Seq_Len). True indicates a position to be scored.
            If None, all positions are considered valid.

    Returns:
        float: The MCRMSE score.
    """
    # 1. Convert inputs to NumPy arrays if they are Tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # 2. Validation
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    # 3. Compute RMSE per column
    # We iterate over the last dimension (channels/targets)
    num_columns = preds.shape[-1]
    column_rmses = []

    for col_idx in range(num_columns):
        # Extract the specific column data: (Batch, Seq_Len)
        pred_col = preds[..., col_idx]
        target_col = targets[..., col_idx]

        if mask is not None:
            # Ensure mask is boolean
            bool_mask = mask.astype(bool)

            # Select only valid elements using the mask
            # This flattens the array to 1D containing only valid entries
            valid_preds = pred_col[bool_mask]
            valid_targets = target_col[bool_mask]
        else:
            valid_preds = pred_col.flatten()
            valid_targets = target_col.flatten()

        # Check if we have data points to score
        if len(valid_preds) == 0:
            # If a column has no valid data points (unlikely in this task),
            # we assign 0.0 error or handle gracefully.
            column_rmses.append(0.0)
            continue

        # Compute MSE for this column
        mse = np.mean((valid_preds - valid_targets) ** 2)

        # Compute RMSE for this column
        rmse = np.sqrt(mse)
        column_rmses.append(rmse)

    # 4. Compute Mean of RMSEs
    mcrmse = np.mean(column_rmses)

    return float(mcrmse)
