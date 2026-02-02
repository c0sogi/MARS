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


class MCRMSEMetric:
    """
    Accumulates Sum of Squared Errors (SSE) and counts to compute the
    Mean Columnwise Root Mean Squared Error (MCRMSE) globally.

    This avoids the bias introduced by averaging RMSEs calculated per batch.
    """

    def __init__(self, scored_indices=None):
        """
        Args:
            scored_indices (list of int, optional): Indices of the target columns to score.
                                                    If None, all columns are scored.
        """
        self.scored_indices = scored_indices
        self.total_sse = None
        self.total_count = 0

    def update(self, preds, targets):
        """
        Updates the metric with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor or np.ndarray): Predictions of shape (Batch, Seq_Len, Channels).
            targets (torch.Tensor or np.ndarray): Ground truth of shape (Batch, Seq_Len, Channels).
        """
        # Convert to numpy if tensors
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Select scored columns if indices are provided
        if self.scored_indices is not None:
            preds = preds[..., self.scored_indices]
            targets = targets[..., self.scored_indices]

        # Initialize total_sse vector if this is the first update
        if self.total_sse is None:
            self.total_sse = np.zeros(preds.shape[-1], dtype=np.float64)

        # Flatten all dimensions except the last (channels) to handle (Batch, Seq, Channels)
        # or (Batch * Seq, Channels) uniformly.
        diff = preds - targets
        diff = diff.reshape(-1, diff.shape[-1])

        # Sum squared errors over the flattened batch/sequence dimension
        batch_sse = np.sum(diff**2, axis=0)

        # Update totals
        self.total_sse += batch_sse
        self.total_count += diff.shape[0]

    def compute(self):
        """
        Computes the final MCRMSE metric based on accumulated data.

        Returns:
            float: The mean columnwise RMSE.
        """
        if self.total_count == 0:
            return 0.0

        # Mean Squared Error per column
        mse = self.total_sse / self.total_count

        # Root Mean Squared Error per column
        rmse = np.sqrt(mse)

        # Mean of RMSEs across columns (MCRMSE)
        mcrmse = np.mean(rmse)

        return mcrmse

    def reset(self):
        """Resets the internal state."""
        self.total_sse = None
        self.total_count = 0


def compute_global_mcrmse(preds, targets, scored_indices=None):
    """
    Computes MCRMSE on the full dataset provided as single arrays/tensors.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions.
        targets (torch.Tensor or np.ndarray): Ground truth.
        scored_indices (list of int, optional): Indices of columns to score.

    Returns:
        float: The calculated MCRMSE.
    """
    metric = MCRMSEMetric(scored_indices)
    metric.update(preds, targets)
    return metric.compute()
