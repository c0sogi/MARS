import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Column-wise Root Mean Squared Error Loss.
    Calculates the loss on all 5 target columns for optimization purposes.
    Slices predictions to match the length of targets (seq_scored=68).
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Model predictions of shape (Batch, Seq_Len_Out, 5).
                   Usually Seq_Len_Out is 107.
            targets: Ground truth values of shape (Batch, Seq_Len_Tgt, 5).
                     Usually Seq_Len_Tgt is 68.
        Returns:
            torch.Tensor: Scalar loss value (mean of RMSE per column).
        """
        # Slice predictions to match target length (first 68 positions)
        seq_len_target = targets.shape[1]
        preds_sliced = preds[:, :seq_len_target, :]

        # Calculate MSE per column (averaging over batch and sequence length)
        # Shape: (5,)
        mse_per_col = torch.mean((preds_sliced - targets) ** 2, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Average RMSE over all columns for the loss
        loss = torch.mean(rmse_per_col)

        return loss


def compute_mcrmse(
    preds: np.ndarray, targets: np.ndarray, scored_only: bool = True
) -> float:
    """
    Calculates the MCRMSE metric on NumPy arrays.

    Args:
        preds: Predictions array of shape (N, Seq_Len, 5).
        targets: Ground truth array of shape (N, Seq_Len_Tgt, 5).
        scored_only: If True, calculates metric only for the 3 scored columns:
                     ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'].
                     If False, calculates for all 5 columns.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Slice predictions to match target length
    seq_len_target = targets.shape[1]
    preds_sliced = preds[:, :seq_len_target, :]

    # Calculate Squared Error
    squared_error = (preds_sliced - targets) ** 2

    # Average over samples and sequence length (axis 0 and 1) -> shape (5,)
    mse_per_col = np.mean(squared_error, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)

    if scored_only:
        # Identify indices of the columns used for scoring
        scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        # Average RMSE only for the scored columns
        final_score = np.mean(rmse_per_col[scored_indices])
    else:
        # Average RMSE for all columns
        final_score = np.mean(rmse_per_col)

    return final_score


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
