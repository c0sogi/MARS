import torch
from library.config import Config


def mcrmse_loss(pred, target):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    Applies strict masking:
    1. Sequence Masking: Only considers the first Config.SEQ_SCORED positions.
    2. Target Masking: Only considers the columns specified in Config.SCORED_COLS_INDICES.

    Args:
        pred (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
        target (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # 1. Slice to valid sequence length (0 to 67)
    # Shape: (Batch, 68, 5)
    pred_seq = pred[:, : Config.SEQ_SCORED, :]
    target_seq = target[:, : Config.SEQ_SCORED, :]

    # 2. Slice to scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Shape: (Batch, 68, 3)
    pred_scored = pred_seq[:, :, Config.SCORED_COLS_INDICES]
    target_scored = target_seq[:, :, Config.SCORED_COLS_INDICES]

    # 3. Compute MSE per column
    # Average over Batch (dim 0) and Sequence (dim 1)
    # Shape: (3,)
    mse_per_col = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 1))

    # 4. Compute RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # 5. Average RMSE across columns to get MCRMSE
    loss = torch.mean(rmse_per_col)

    return loss


class GlobalMCRMSE:
    """
    Accumulates predictions and targets to compute the global MCRMSE
    over the entire validation set.

    This avoids the bias introduced by averaging RMSEs calculated on small batches.
    """

    def __init__(self):
        self.scored_indices = Config.SCORED_COLS_INDICES
        self.seq_scored = Config.SEQ_SCORED
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sum_squared_errors = None
        self.total_count = 0

    def update(self, pred, target):
        """
        Updates the metric with a new batch of predictions and targets.

        Args:
            pred (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
            target (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).
        """
        # Ensure inputs are on the same device
        device = pred.device

        # 1. Slice to valid sequence length and scored columns
        # Shape: (Batch, 68, 3)
        p = pred[:, : self.seq_scored, self.scored_indices]
        t = target[:, : self.seq_scored, self.scored_indices]

        # 2. Calculate squared errors
        squared_errors = (p - t) ** 2

        # 3. Sum squared errors over batch and sequence dimensions
        # Shape: (3,)
        batch_sse = torch.sum(squared_errors, dim=(0, 1))

        # 4. Count total elements contributing to the sum (Batch * Seq_Scored)
        batch_count = p.shape[0] * p.shape[1]

        # 5. Accumulate
        if self.sum_squared_errors is None:
            self.sum_squared_errors = torch.zeros_like(batch_sse, device=device)

        # Detach to prevent memory leaks from graph accumulation
        self.sum_squared_errors += batch_sse.detach()
        self.total_count += batch_count

    def compute(self):
        """
        Computes the final global MCRMSE.

        Returns:
            float: The MCRMSE value.
        """
        if self.total_count == 0:
            return 0.0

        # Calculate MSE per column
        mse_per_col = self.sum_squared_errors / self.total_count

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Calculate mean across columns
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse.item()
