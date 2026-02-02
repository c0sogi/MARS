import torch
import torch.nn.functional as F


def masked_mse_loss(preds, targets, mask):
    """
    Computes Mean Squared Error (MSE) loss, considering only valid positions defined by the mask.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 3).
        targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 3).
        mask (torch.Tensor): Binary mask of shape (Batch, Seq_Len), where 1 indicates a scored position.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Expand mask to align with the channel dimension of targets/preds
    # Shape: (Batch, Seq_Len) -> (Batch, Seq_Len, 1)
    mask_expanded = mask.unsqueeze(-1)

    # Compute element-wise squared errors
    loss = F.mse_loss(preds, targets, reduction="none")

    # Zero out errors at invalid positions
    masked_loss = loss * mask_expanded

    # Compute the mean loss over valid elements
    # Total valid elements = (Number of valid positions) * (Number of channels)
    # We multiply mask sum by preds.shape[-1] (which is 3)
    num_valid_elements = mask_expanded.sum() * preds.shape[-1]

    # Avoid division by zero
    if num_valid_elements == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    return masked_loss.sum() / num_valid_elements


def mcrmse(preds, targets, mask):
    """
    Computes Mean Columnwise Root Mean Squared Error (MCRMSE).

    Steps:
    1. Compute MSE for each of the 3 columns separately on masked positions.
    2. Take the square root to get RMSE for each column.
    3. Average the 3 RMSE values.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 3).
        targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 3).
        mask (torch.Tensor): Binary mask of shape (Batch, Seq_Len).

    Returns:
        torch.Tensor: Scalar MCRMSE value.
    """
    # Shape: (Batch, Seq_Len, 1)
    mask_expanded = mask.unsqueeze(-1)

    # Squared errors: (Batch, Seq_Len, 3)
    squared_errors = (preds - targets) ** 2

    # Apply mask: (Batch, Seq_Len, 3)
    masked_squared_errors = squared_errors * mask_expanded

    # Sum errors across Batch and Sequence dimensions for each column
    # Result shape: (3,)
    sum_errors_per_col = masked_squared_errors.sum(dim=(0, 1))

    # Count valid positions. Since mask is shared across columns, count is same for all.
    # mask_expanded.sum(dim=(0, 1)) returns shape (1,)
    num_valid_per_col = mask_expanded.sum(dim=(0, 1))

    # Avoid division by zero
    num_valid_per_col = torch.clamp(num_valid_per_col, min=1.0)

    # MSE per column: (3,)
    mse_per_col = sum_errors_per_col / num_valid_per_col

    # RMSE per column: (3,)
    rmse_per_col = torch.sqrt(mse_per_col)

    # Average RMSE across the 3 columns
    return rmse_per_col.mean()
