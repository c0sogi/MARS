import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask_len: int = 68,
    scored_indices: list = None,
) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each scored column separately and then
    takes the average of those RMSEs. It strictly respects the scored sequence length
    and the specific columns designated for scoring.

    Args:
        preds (torch.Tensor): Predicted values of shape (Batch, Length, Channels).
        targets (torch.Tensor): Ground truth values of shape (Batch, Length, Channels).
        mask_len (int): The number of positions from the start of the sequence to score.
                        Defaults to 68 (standard for this dataset).
        scored_indices (list, optional): List of integer indices corresponding to the columns
                                         that should be included in the metric.
                                         Defaults to [0, 1, 3] corresponding to:
                                         [reactivity, deg_Mg_pH10, deg_Mg_50C].

    Returns:
        torch.Tensor: A scalar tensor containing the MCRMSE loss.
    """
    if scored_indices is None:
        # Default scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Based on column order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        scored_indices = [0, 1, 3]

    # Slice the tensors to the valid scored length
    # Shape becomes: (Batch, mask_len, Channels)
    valid_preds = preds[:, :mask_len, :]
    valid_targets = targets[:, :mask_len, :]

    # Select only the columns relevant for the metric
    # Shape becomes: (Batch, mask_len, len(scored_indices))
    selected_preds = valid_preds[:, :, scored_indices]
    selected_targets = valid_targets[:, :, scored_indices]

    # Compute MSE per column (averaging over batch and sequence length dimensions)
    # Result shape: (len(scored_indices),)
    mse = torch.mean((selected_preds - selected_targets) ** 2, dim=(0, 1))

    # Compute RMSE per column
    rmse = torch.sqrt(mse)

    # Compute Mean of RMSEs
    mcrmse = torch.mean(rmse)

    return mcrmse
