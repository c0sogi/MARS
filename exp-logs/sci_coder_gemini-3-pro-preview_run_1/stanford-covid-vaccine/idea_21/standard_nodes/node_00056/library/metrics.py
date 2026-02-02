import torch
from library.loss import compute_mcrmse


def calculate_mcrmse(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function calculates the RMSE for each of the 3 scored columns (reactivity,
    deg_Mg_pH10, deg_Mg_50C) separately, considering only the valid positions
    indicated by the mask. It then returns the average of these 3 RMSE values.

    This approach corrects the 'Mean of Sqrts' artifact by ensuring averaging happens
    after the square root operation for each column.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 3).
        targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 3).
        mask (torch.Tensor): Boolean mask of shape (Batch, Seq_Len), where True
                             indicates a valid position to be scored.

    Returns:
        torch.Tensor: The scalar MCRMSE score.
    """
    # Delegate to the provided library implementation to avoid code duplication
    return compute_mcrmse(preds, targets, mask)
