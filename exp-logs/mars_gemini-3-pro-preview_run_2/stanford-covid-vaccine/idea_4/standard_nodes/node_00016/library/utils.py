import os
import random
import numpy as np
import torch
import torch.nn as nn
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function calculates the RMSE for each target column separately
    and then averages the RMSE values. It handles the discrepancy between
    prediction length (107) and target length (68) by slicing the predictions.
    """

    def __init__(self, scored_only=True):
        super().__init__()
        self.scored_only = scored_only
        self.seq_scored = config.SEQ_SCORED
        self.scored_indices = config.SCORED_INDICES

    def forward(self, preds, targets):
        """
        Calculates the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len_Pred, Channels).
                                  Typically (B, 107, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len_Target, Channels).
                                    Typically (B, 68, 5).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Slice predictions to match the length of the targets
        # The competition only scores the first 68 positions
        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        # Calculate MSE for each column: mean over Batch (dim 0) and Sequence (dim 1)
        # Result shape: (Channels,)
        mse = torch.mean((preds - targets) ** 2, dim=(0, 1))

        # Calculate RMSE for each column
        rmse = torch.sqrt(mse)

        # Filter for only the scored columns if requested
        if self.scored_only:
            rmse = rmse[self.scored_indices]

        # Calculate Mean of RMSEs across all columns
        mcrmse = torch.mean(rmse)

        return mcrmse
