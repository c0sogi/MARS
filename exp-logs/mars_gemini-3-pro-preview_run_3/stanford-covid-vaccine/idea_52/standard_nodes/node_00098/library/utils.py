import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_loss(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss for training.
    This function computes the error on all 5 targets to maximize signal utilization
    during the training phase.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
        targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Calculate Mean Squared Error (MSE) for each of the 5 columns
    # Averaging over batch (dim 0) and sequence (dim 1) dimensions
    colwise_mse = torch.mean((preds - targets) ** 2, dim=(0, 1))

    # Calculate Root Mean Squared Error (RMSE) for each column
    colwise_rmse = torch.sqrt(colwise_mse)

    # Calculate the mean of the column RMSEs
    loss = torch.mean(colwise_rmse)

    return loss


def calculate_metric(preds, targets):
    """
    Calculates the MCRMSE metric for validation, strictly adhering to the competition's
    scoring protocol (Metric Integrity).

    Logic:
    1. Slices predictions to the first 68 positions (Config.SEQ_SCORED).
    2. Selects only the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    3. Computes MCRMSE on this subset.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions of shape (N_samples, 107, 5).
        targets (np.ndarray or torch.Tensor): Ground truth of shape (N_samples, 68, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # 1. Slice predictions to the first 68 positions
    # Note: Targets are expected to be already sliced or loaded as 68 length
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]

    # Validation check for shapes
    if preds_sliced.shape[1] != targets.shape[1]:
        raise ValueError(
            f"Sequence length mismatch in metric calculation. "
            f"Preds sliced: {preds_sliced.shape[1]}, Targets: {targets.shape[1]}"
        )

    # 2. Select scored columns
    # Indices based on: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    # Scored: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]

    preds_selected = preds_sliced[:, :, scored_indices]
    targets_selected = targets[:, :, scored_indices]

    # 3. Compute MCRMSE
    # Flatten batch and sequence dimensions to compute global column-wise MSE
    preds_flat = preds_selected.reshape(-1, len(scored_indices))
    targets_flat = targets_selected.reshape(-1, len(scored_indices))

    # Mean Squared Error per column
    mse = np.mean((preds_flat - targets_flat) ** 2, axis=0)

    # Root Mean Squared Error per column
    rmse = np.sqrt(mse)

    # Mean of RMSEs
    mcrmse = np.mean(rmse)

    return float(mcrmse)
