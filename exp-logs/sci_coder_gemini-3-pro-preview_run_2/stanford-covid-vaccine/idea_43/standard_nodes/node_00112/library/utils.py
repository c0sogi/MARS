import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


class MetricTracker:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE)
    accumulated over batches.

    It specifically tracks the scored columns: reactivity, deg_Mg_pH10, and deg_Mg_50C.
    """

    def __init__(self):
        # Indices corresponding to ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        # within the full target list ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        self.scored_indices = [0, 1, 3]
        self.reset()

    def reset(self):
        """Resets the internal state."""
        # Sum of squared errors for each of the 3 scored columns
        self.sum_squared_errors = np.zeros(len(self.scored_indices), dtype=np.float64)
        # Count of elements for each column
        self.counts = np.zeros(len(self.scored_indices), dtype=np.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates the metric with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5) or (Batch, Seq_Len, 3).
                                  If 5 channels, it will be sliced.
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).
        """
        # Detach and move to CPU numpy
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # Handle shapes. If preds has 5 channels, slice it.
        # If preds already has 3 channels (e.g. model output limited), assume they are the correct ones.
        # However, the Config implies the model outputs 5 channels to match target structure during training.

        # Select scored columns from targets
        # targets shape: (B, L, 5) -> (B, L, 3)
        targets_scored = targets[:, :, self.scored_indices]

        if preds.shape[-1] == 5:
            preds_scored = preds[:, :, self.scored_indices]
        else:
            # Fallback if model architecture was changed to output only 3
            # But based on standard practice here, we assume 5 outputs.
            preds_scored = preds

        # Compute squared errors
        squared_errors = (preds_scored - targets_scored) ** 2

        # Sum over batch and sequence dimensions for each column
        # squared_errors shape: (B, L, 3) -> sum -> (3,)
        batch_sse = np.sum(squared_errors, axis=(0, 1))

        # Count elements (Batch * Seq_Len)
        batch_count = preds_scored.shape[0] * preds_scored.shape[1]

        self.sum_squared_errors += batch_sse
        self.counts += batch_count

    def compute(self):
        """
        Computes the final MCRMSE metric.

        Returns:
            float: The mean of the RMSEs of the scored columns.
        """
        # Avoid division by zero
        safe_counts = np.maximum(self.counts, 1.0)

        # MSE per column
        mse = self.sum_squared_errors / safe_counts

        # RMSE per column
        rmse = np.sqrt(mse)

        # MCRMSE is the mean of the RMSEs
        mcrmse = np.mean(rmse)

        return mcrmse
