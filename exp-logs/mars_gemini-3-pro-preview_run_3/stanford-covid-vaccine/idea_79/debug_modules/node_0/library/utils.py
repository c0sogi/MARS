import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    Calculates the RMSE for each target column separately and then averages them.
    Handles slicing of predictions to match the target sequence length (seq_scored).
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets).
                                   Usually (B, 107, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Scored, Num_Targets).
                                    Usually (B, 68, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice inputs to match target length (first 68 positions)
        # targets.shape[1] should be Config.SEQ_SCORED (68)
        seq_scored = targets.shape[1]
        inputs_sliced = inputs[:, :seq_scored, :]

        # Calculate MSE per column (averaging over batch and sequence dimensions)
        # Shape: (Num_Targets,)
        mse = torch.mean((inputs_sliced - targets) ** 2, dim=(0, 1))

        # Calculate RMSE per column
        rmse = torch.sqrt(mse)

        # Average RMSE across all columns
        loss = torch.mean(rmse)

        return loss


def calculate_metric(preds, targets):
    """
    Calculates the competition metric: MCRMSE on specific scored columns.

    Args:
        preds (np.ndarray): Predictions of shape (N_Samples, Seq_Len, 5).
        targets (np.ndarray): Ground truth of shape (N_Samples, Seq_Scored, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # 1. Slice predictions to the scored sequence length (68)
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]

    # 2. Select only the columns used for scoring
    # Config.SCORED_TARGET_INDICES = [0, 1, 3] corresponding to:
    # reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = Config.SCORED_TARGET_INDICES

    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets[:, :, scored_indices]

    # 3. Compute RMSE for each column
    # Flatten batch and sequence dimensions to compute global RMSE per column
    # Shape becomes (N_Samples * Seq_Scored, 3)
    preds_flat = preds_filtered.reshape(-1, len(scored_indices))
    targets_flat = targets_filtered.reshape(-1, len(scored_indices))

    # MSE per column
    mse = np.mean((preds_flat - targets_flat) ** 2, axis=0)

    # RMSE per column
    rmse = np.sqrt(mse)

    # 4. Mean of RMSEs
    mcrmse = np.mean(rmse)

    return mcrmse
