import os
import random
import numpy as np
import torch


def set_seed(seed=42):
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


class GlobalMCRMSE:
    """
    Accumulates Sum of Squared Errors (SSE) and counts over the entire validation set
    to compute the Global Mean Columnwise Root Mean Squared Error (MCRMSE).

    This avoids the statistical bias introduced by averaging RMSEs calculated per batch.
    """

    def __init__(self, scored_indices=None, device="cpu"):
        """
        Args:
            scored_indices (list of int): Indices of the columns to include in the final metric.
                                          If None, defaults to [0, 1, 3] corresponding to
                                          ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'].
            device (str): Device to store the accumulators on.
        """
        # Default to the competition scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
        # based on ALL_TARGETS order: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
        if scored_indices is None:
            self.scored_indices = [0, 1, 3]
        else:
            self.scored_indices = scored_indices

        self.device = device
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # We assume 5 target columns based on the dataset spec
        self.total_sse = torch.zeros(5, dtype=torch.float64, device=self.device)
        self.total_count = torch.zeros(5, dtype=torch.float64, device=self.device)

    def update(self, preds, targets, mask):
        """
        Updates the accumulators with a new batch of predictions.

        Args:
            preds (torch.Tensor): Predictions of shape (B, L, 5).
            targets (torch.Tensor): Ground truth of shape (B, L, 5).
            mask (torch.Tensor): Mask of shape (B, L, 5) indicating valid positions.
        """
        # Ensure inputs are on the correct device
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        mask = mask.to(self.device)

        # Calculate squared errors
        squared_errors = (preds - targets) ** 2

        # Apply mask and sum over batch and sequence dimensions
        # Result shape: (5,)
        masked_sse = (squared_errors * mask).sum(dim=(0, 1))
        valid_counts = mask.sum(dim=(0, 1))

        self.total_sse += masked_sse.double()
        self.total_count += valid_counts.double()

    def compute(self):
        """
        Computes the final MCRMSE metric based on accumulated data.

        Returns:
            float: The mean columnwise RMSE over the scored columns.
        """
        rmses = []
        for idx in self.scored_indices:
            count = self.total_count[idx]
            sse = self.total_sse[idx]

            if count > 0:
                # RMSE for this specific column
                rmse = torch.sqrt(sse / count)
                rmses.append(rmse)

        if not rmses:
            return 0.0

        # Mean of the RMSEs of the scored columns
        final_metric = torch.stack(rmses).mean().item()
        return final_metric
