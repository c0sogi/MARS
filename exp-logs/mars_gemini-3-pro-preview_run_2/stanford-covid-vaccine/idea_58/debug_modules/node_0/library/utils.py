import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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

    This loss function calculates the RMSE for each scored column separately
    and then averages them. It explicitly handles masking by only considering
    the first `Config.SCORED_LEN` sequence positions and the specific columns
    defined in `Config.SCORED_COLS`.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # Determine the indices of the columns that contribute to the score
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions of shape (Batch, Seq_Len, Num_Targets)
            targets: Ground truth of shape (Batch, Seq_Len, Num_Targets)

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice to the scored sequence length (e.g., first 68 positions)
        #    Positions beyond SCORED_LEN are not used for scoring.
        inputs_sliced = inputs[:, : Config.SCORED_LEN, :]
        targets_sliced = targets[:, : Config.SCORED_LEN, :]

        # 2. Select only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        inputs_scored = inputs_sliced[:, :, self.scored_indices]
        targets_scored = targets_sliced[:, :, self.scored_indices]

        # 3. Compute Squared Error per element
        squared_error = (inputs_scored - targets_scored) ** 2

        # 4. Compute Mean Squared Error per column
        #    Average over Batch (dim 0) and Sequence (dim 1) dimensions
        mse_per_column = torch.mean(squared_error, dim=(0, 1))

        # 5. Compute RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 6. Average RMSEs across the scored columns to get MCRMSE
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse


class GlobalMetrics:
    """
    Accumulates statistics over the entire validation set to compute the
    global MCRMSE without batch-averaging bias.
    """

    def __init__(self):
        # Determine the indices of the columns that contribute to the score
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # Accumulator for Sum of Squared Errors per scored column
        self.total_sse = torch.zeros(len(self.scored_indices), device=Config.DEVICE)
        # Accumulator for total number of elements per column
        self.total_count = 0.0

    def update(self, preds, targets):
        """
        Updates the accumulators with predictions and targets from a single batch.

        Args:
            preds: Predictions of shape (Batch, Seq_Len, Num_Targets)
            targets: Ground truth of shape (Batch, Seq_Len, Num_Targets)
        """
        with torch.no_grad():
            # Ensure tensors are on the correct device
            preds = preds.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # 1. Slice to the scored sequence length
            preds_sliced = preds[:, : Config.SCORED_LEN, :]
            targets_sliced = targets[:, : Config.SCORED_LEN, :]

            # 2. Select only the scored columns
            preds_scored = preds_sliced[:, :, self.scored_indices]
            targets_scored = targets_sliced[:, :, self.scored_indices]

            # 3. Compute Sum of Squared Errors (SSE) per column for this batch
            #    Sum over Batch (dim 0) and Sequence (dim 1)
            batch_sse = torch.sum((preds_scored - targets_scored) ** 2, dim=(0, 1))

            # 4. Update total SSE
            self.total_sse += batch_sse

            # 5. Update total count
            #    Count = Batch_Size * Scored_Sequence_Length
            batch_size = preds.shape[0]
            self.total_count += batch_size * Config.SCORED_LEN

    def compute(self):
        """
        Computes the final global MCRMSE metric based on accumulated statistics.

        Returns:
            float: The global MCRMSE value.
        """
        # 1. Compute MSE per column (Total SSE / Total Count)
        mse_per_column = self.total_sse / self.total_count

        # 2. Compute RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 3. Compute Mean of RMSEs
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse.item()
