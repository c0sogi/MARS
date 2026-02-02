import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSE:
    """
    Accumulates Mean Columnwise Root Mean Squared Error (MCRMSE) globally.

    This class tracks the Sum of Squared Errors (SSE) and the count of valid predictions
    for each column separately. This allows for the calculation of the 'Correct Global RMSE'
    by computing sqrt(sum(error^2) / count) over the entire dataset, rather than
    averaging the RMSE of individual batches.
    """

    def __init__(self, scored_indices=None, seq_scored=None):
        """
        Args:
            scored_indices (list, optional): Indices of the columns to include in the metric.
                                             Defaults to [0, 1, 3] corresponding to
                                             [reactivity, deg_Mg_pH10, deg_Mg_50C].
            seq_scored (int, optional): The number of positions from the start of the sequence
                                        to score. Defaults to Config.PRED_LEN (68).
        """
        # Default to the competition scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Assuming order: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
        if scored_indices is None:
            self.scored_indices = [0, 1, 3]
        else:
            self.scored_indices = scored_indices

        if seq_scored is None:
            self.seq_scored = Config.PRED_LEN
        else:
            self.seq_scored = seq_scored

        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sse = {idx: 0.0 for idx in self.scored_indices}
        self.count = {idx: 0 for idx in self.scored_indices}

    def update(self, preds, targets, mask=None):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Channels).
            mask (torch.Tensor, optional): Boolean mask of shape (Batch, Seq_Len).
                                           If None, assumes the first `seq_scored` positions are valid.
        """
        # Detach and move to CPU/Numpy to avoid GPU memory accumulation
        preds = preds.detach().float()
        targets = targets.detach().float()

        # Slice to the scored sequence length if no mask is provided
        # This aligns with the competition rule of scoring the first 68 bases
        if mask is None:
            # Create a mask for the first seq_scored positions
            # preds shape: (B, L, C)
            B, L, C = preds.shape
            # Slice tensors directly to save computation
            eff_len = min(L, self.seq_scored)
            preds = preds[:, :eff_len, :]
            targets = targets[:, :eff_len, :]
            # Create a full mask for the sliced region
            mask_np = np.ones((B, eff_len), dtype=bool)
        else:
            mask = mask.detach().bool().cpu().numpy()
            mask_np = mask

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for idx in self.scored_indices:
            # Extract specific column
            p = preds_np[:, :, idx]
            t = targets_np[:, :, idx]

            # Calculate squared errors
            squared_errors = (p - t) ** 2

            # Apply mask
            if mask_np is not None:
                # Ensure mask shape matches if slicing happened differently (should match here)
                valid_errors = squared_errors[mask_np]
            else:
                valid_errors = squared_errors.flatten()

            self.sse[idx] += np.sum(valid_errors)
            self.count[idx] += valid_errors.size

    def compute(self):
        """
        Computes the final MCRMSE score.

        Returns:
            float: The mean of the RMSEs of the scored columns.
        """
        rmses = []
        for idx in self.scored_indices:
            total_sse = self.sse[idx]
            total_count = self.count[idx]

            if total_count > 0:
                mse = total_sse / total_count
                rmse = np.sqrt(mse)
                rmses.append(rmse)
            else:
                rmses.append(0.0)

        if not rmses:
            return 0.0

        return np.mean(rmses)
