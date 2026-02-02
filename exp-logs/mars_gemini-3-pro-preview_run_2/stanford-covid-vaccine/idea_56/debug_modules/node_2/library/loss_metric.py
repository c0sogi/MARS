import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Computes the Mean Column-wise Root Mean Squared Error (MCRMSE) loss.
    This loss function strictly adheres to the competition metric by:
    1. Truncating predictions to the scored sequence length (68).
    2. Filtering only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    """

    def __init__(self):
        super().__init__()
        # Identify indices of columns that are used for scoring
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        # Register indices as a buffer to ensure they move to the correct device
        self.register_buffer(
            "scored_idxs", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len_Pred, 5)
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len_Target, 5)

        Returns:
            torch.Tensor: Scalar MCRMSE loss
        """
        # 1. Align sequence length
        # Predictions (107) -> Targets (68)
        seq_len_target = targets.shape[1]
        preds_sliced = preds[:, :seq_len_target, :]

        # 2. Select scored columns
        # Shape: (Batch, Seq_Len_Target, 3)
        preds_scored = torch.index_select(preds_sliced, 2, self.scored_idxs)
        targets_scored = torch.index_select(targets, 2, self.scored_idxs)

        # 3. Compute MSE per column
        # Average over Batch and Sequence dimensions
        mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Average RMSEs to get MCRMSE
        mcrmse = torch.mean(rmse)

        return mcrmse


class GlobalMCRMSE:
    """
    Accumulates metrics over the entire validation set to compute the
    Global MCRMSE. This avoids the bias introduced by averaging
    RMSE scores calculated on small individual batches.
    """

    def __init__(self):
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sse = None  # Sum of Squared Errors per column
        self.count = 0  # Total number of valid elements per column

    def update(self, preds, targets):
        """
        Updates the running statistics with a new batch of data.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5)
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len_Target, 5)
        """
        # Detach from graph to save memory
        preds = preds.detach()
        targets = targets.detach()

        # 1. Align sequence length
        seq_len_target = targets.shape[1]
        preds_sliced = preds[:, :seq_len_target, :]

        # 2. Select scored columns
        # We manually slice using the list of indices
        preds_scored = preds_sliced[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # 3. Compute Squared Errors
        sq_diff = (preds_scored - targets_scored) ** 2

        # 4. Accumulate Sum of Squared Errors (SSE)
        # Sum over Batch and Sequence dimensions -> Shape: (Num_Scored_Cols,)
        batch_sse = torch.sum(sq_diff, dim=(0, 1))

        # 5. Accumulate Count
        # Total elements = Batch_Size * Seq_Len_Target
        batch_count = preds_scored.shape[0] * preds_scored.shape[1]

        if self.sse is None:
            self.sse = torch.zeros_like(batch_sse)

        self.sse += batch_sse
        self.count += batch_count

    def compute(self):
        """
        Computes the final MCRMSE metric based on accumulated statistics.

        Returns:
            float: The global MCRMSE score.
        """
        if self.count == 0:
            return 0.0

        # Global MSE per column
        mse = self.sse / self.count

        # Global RMSE per column
        rmse = torch.sqrt(mse)

        # Mean of RMSEs
        mcrmse = torch.mean(rmse)

        return mcrmse.item()
