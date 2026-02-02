import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This loss function handles:
    1. Slicing sequences to the scored length (Config.PRED_LEN).
    2. Selecting specific columns for validation scoring vs training.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # Determine indices for the scored targets based on Config
        # TARGET_COLS: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_TARGETS: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.all_targets = Config.TARGET_COLS
        self.scored_targets = Config.SCORED_TARGETS

        # Create a list of indices corresponding to the scored targets
        self.scored_indices = [self.all_targets.index(t) for t in self.scored_targets]

        # Ensure indices are on the correct device if needed (though usually used for slicing)
        # We keep them as a python list for slicing logic.

    def forward(self, preds, targets, scoring_only=False):
        """
        Computes the MCRMSE.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).
            scoring_only (bool): If True, calculates metric only on the 3 scored columns.
                                 If False, calculates on all 5 columns.

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # 1. Slice to the scored sequence length (first 68 positions)
        # The metric is only evaluated on the first seq_scored positions.
        preds_sliced = preds[:, : Config.PRED_LEN, :]
        targets_sliced = targets[:, : Config.PRED_LEN, :]

        # 2. Filter columns if this is for official scoring validation
        if scoring_only:
            preds_sliced = preds_sliced[:, :, self.scored_indices]
            targets_sliced = targets_sliced[:, :, self.scored_indices]

        # 3. Calculate MSE per column
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # Result shape: (Num_Columns,)
        mse_per_column = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(0, 1))

        # 4. Calculate RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 5. Average RMSE across all columns to get MCRMSE
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse
