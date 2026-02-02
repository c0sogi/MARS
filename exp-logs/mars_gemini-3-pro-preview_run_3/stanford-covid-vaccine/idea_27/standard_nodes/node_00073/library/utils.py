import torch
import numpy as np
import random
import os
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MCRMSE:
    """
    Accumulates predictions and targets to compute the MCRMSE metric globally.
    Handles slicing of predictions to the scored sequence length and ensures
    metrics are calculated over the entire dataset aggregation to avoid batch-size bias.
    """

    def __init__(self):
        self.preds_list = []
        self.targets_list = []

    def update(self, preds, targets):
        """
        Update the metric tracker with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Scored, 5) or (Batch, Seq_Len, 5).
        """
        # Ensure inputs are on CPU and detached to save GPU memory
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()

        # Slice predictions to the scored length (first 68 positions)
        # Config.PRED_LEN is 68
        if preds.shape[1] > Config.PRED_LEN:
            preds_sliced = preds[:, : Config.PRED_LEN, :]
        else:
            preds_sliced = preds

        # Handle targets. If targets are provided as (B, 68, 5), use as is.
        # If targets are (B, 107, 5), slice them to match predictions.
        if targets.shape[1] > Config.PRED_LEN:
            targets_sliced = targets[:, : Config.PRED_LEN, :]
        else:
            targets_sliced = targets

        self.preds_list.append(preds_sliced)
        self.targets_list.append(targets_sliced)

    def compute(self):
        """
        Compute the global MCRMSE over all accumulated batches.

        Formula:
        MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

        Returns:
            float: The MCRMSE score.
        """
        if not self.preds_list:
            return 0.0

        # Concatenate all batches to perform global aggregation
        # Shape: (Total_Samples, 68, 5)
        all_preds = torch.cat(self.preds_list, dim=0)
        all_targets = torch.cat(self.targets_list, dim=0)

        # Calculate MSE per column (averaging over samples and sequence positions)
        # dim=(0, 1) aggregates over Batch and Sequence dimensions, leaving the 5 Target columns
        mse = torch.mean((all_preds - all_targets) ** 2, dim=(0, 1))

        # RMSE per column
        rmse = torch.sqrt(mse)

        # Filter for scored columns only
        if hasattr(Config, "SCORED_COLS_INDICES"):
            rmse = rmse[Config.SCORED_COLS_INDICES]

        # Mean of RMSEs (MCRMSE) across the 5 target columns
        mcrmse = torch.mean(rmse)

        return mcrmse.item()

    def reset(self):
        """
        Reset the accumulator for the next epoch.
        """
        self.preds_list = []
        self.targets_list = []
