import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior in CuDNN.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSE_Metric:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored columns.

    This class accumulates Squared Errors and counts globally over the entire validation set
    before computing the square root. This avoids the bias introduced by averaging
    RMSEs calculated on small batches.
    """

    def __init__(self):
        # Retrieve column definitions from Config
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS

        # Determine indices of the columns that are actually scored
        # e.g., if TARGET_COLS has 5 items and SCORED_COLS has indices 0, 1, 3
        self.scored_indices = [
            i for i, col in enumerate(self.target_cols) if col in self.scored_cols
        ]

        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # Accumulators for Sum of Squared Errors and Counts
        # Using double precision (float64) for accumulation stability
        self.total_sse = torch.zeros(len(self.scored_indices), dtype=torch.float64)
        self.total_count = torch.zeros(len(self.scored_indices), dtype=torch.float64)

    def update(self, preds, targets, mask=None):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predictions tensor. Shape (Batch, Seq, Channels) or (N, Channels).
            targets (torch.Tensor): Ground truth tensor. Shape (Batch, Seq, Channels) or (N, Channels).
            mask (torch.Tensor, optional): Boolean mask indicating valid positions.
                                           Shape (Batch, Seq) or (N,).
        """
        # Detach and move to CPU for accumulation
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # Filter predictions and targets to include only the scored columns
        preds_scored = preds[..., self.scored_indices]
        targets_scored = targets[..., self.scored_indices]

        # Apply masking if provided
        if mask is not None:
            mask = mask.detach().cpu()
            # Ensure mask is boolean
            mask = mask > 0

            # Flatten tensors to (N_valid, C)
            if preds_scored.dim() == 3 and mask.dim() == 2:
                # Standard (Batch, Seq, Channels) case
                B, L, C = preds_scored.shape
                preds_flat = preds_scored.reshape(-1, C)
                targets_flat = targets_scored.reshape(-1, C)
                mask_flat = mask.reshape(-1)

                valid_preds = preds_flat[mask_flat]
                valid_targets = targets_flat[mask_flat]
            else:
                # Handle cases where inputs might already be flattened or different shape
                # This assumes mask shape broadcasts or matches preds shape prefix
                valid_preds = preds_scored[mask]
                valid_targets = targets_scored[mask]
        else:
            # No mask provided, use all data
            valid_preds = preds_scored.reshape(-1, len(self.scored_indices))
            valid_targets = targets_scored.reshape(-1, len(self.scored_indices))

        if valid_preds.numel() == 0:
            return

        # Compute Squared Errors: (y - y_hat)^2
        squared_errors = (valid_preds - valid_targets) ** 2

        # Accumulate SSE per column
        self.total_sse += torch.sum(squared_errors, dim=0).double()

        # Accumulate Count per column
        batch_count = valid_preds.shape[0]
        self.total_count += batch_count

    def compute(self):
        """
        Computes the final MCRMSE metric based on accumulated data.

        Returns:
            float: The global MCRMSE score.
        """
        # Avoid division by zero
        safe_count = torch.clamp(self.total_count, min=1.0)

        # RMSE_j = sqrt( SSE_j / N_j )
        rmse_per_col = torch.sqrt(self.total_sse / safe_count)

        # MCRMSE = Mean( RMSE_j )
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse.item()

    def compute_detailed(self):
        """
        Returns the MCRMSE score and a dictionary of RMSE per column.
        Useful for logging specific degradation channel performance.
        """
        safe_count = torch.clamp(self.total_count, min=1.0)
        rmse_per_col = torch.sqrt(self.total_sse / safe_count)
        mcrmse = torch.mean(rmse_per_col).item()

        details = {col: val.item() for col, val in zip(self.scored_cols, rmse_per_col)}
        return mcrmse, details
