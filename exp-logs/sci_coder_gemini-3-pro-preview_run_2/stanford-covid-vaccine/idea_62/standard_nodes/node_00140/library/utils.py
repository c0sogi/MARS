import ast
import numpy as np
import torch
import torch.nn as nn
from library.config import SCORED_LEN


def parse_list_column(x):
    """
    Parses a stringified list (e.g., '[0.1, 0.2, ...]') into a numpy array.
    Handles potential parsing errors or empty strings by returning an empty array.

    Args:
        x (str): The string representation of the list.

    Returns:
        np.ndarray: A float32 numpy array of the parsed values.
    """
    try:
        # ast.literal_eval is safer than eval
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except (ValueError, SyntaxError, TypeError):
        return np.array([], dtype=np.float32)


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate a partner index map.

    Args:
        structure (str): A string containing '(', ')', and '.' characters representing
                         the secondary structure of the RNA.

    Returns:
        np.ndarray: An integer array of shape (len(structure),).
                    arr[i] contains the index of the base paired with base i.
                    If base i is unpaired, arr[i] is -1.
    """
    seq_len = len(structure)
    partner_indices = np.full(seq_len, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i

    return partner_indices


class MCRMSELoss(nn.Module):
    """
    Differentiable MCRMSE (Mean Columnwise Root Mean Squared Error) Loss.
    Used for training the model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mask=None):
        """
        Computes the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predicted values of shape (Batch, Seq, Channels).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq, Channels).
            mask (torch.Tensor, optional): Boolean mask of shape (Batch, Seq) indicating
                                           valid positions (True) vs padded/unscored (False).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        squared_diff = (preds - targets) ** 2

        if mask is not None:
            # Expand mask to match channel dimension: (B, S) -> (B, S, 1) -> (B, S, C)
            mask_expanded = mask.unsqueeze(-1).expand_as(squared_diff)

            # Apply mask
            squared_diff = squared_diff * mask_expanded

            # Count valid positions (n)
            # We assume the mask is consistent across channels for a given sequence position
            n_valid = mask.sum()

            if n_valid == 0:
                return torch.tensor(0.0, device=preds.device, requires_grad=True)

            # Sum squared errors over batch and sequence dimensions
            sse_per_col = squared_diff.sum(dim=(0, 1))

            # Mean Squared Error per column
            mse_per_col = sse_per_col / n_valid

        else:
            # Standard mean over batch and sequence
            mse_per_col = squared_diff.mean(dim=(0, 1))

        # RMSE per column (add epsilon for numerical stability)
        rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

        # Mean of RMSEs across columns
        return rmse_per_col.mean()


class MCRMSE:
    """
    Accumulates statistics to compute the global MCRMSE metric over the entire validation set.
    This ensures the metric is calculated as:
    mean(sqrt(sum(error^2) / total_count))
    rather than averaging batch-wise RMSEs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sse_per_col = None
        self.count = 0

    def update(self, preds, targets, mask=None):
        """
        Updates the running statistics with a new batch of predictions.

        Args:
            preds (np.ndarray or torch.Tensor): Predictions (Batch, Seq, Channels).
            targets (np.ndarray or torch.Tensor): Ground truth (Batch, Seq, Channels).
            mask (np.ndarray or torch.Tensor, optional): Valid position mask (Batch, Seq).
        """
        # Convert tensors to numpy if necessary
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # Flatten batch and sequence dimensions
        # Shape becomes (N_samples, Channels) where N_samples = Batch * Seq
        if preds.ndim == 3:
            B, S, C = preds.shape
            preds = preds.reshape(B * S, C)
            targets = targets.reshape(B * S, C)
            if mask is not None:
                mask = mask.reshape(B * S)

        # Filter by mask if provided
        if mask is not None:
            valid_indices = mask.astype(bool)
            preds = preds[valid_indices]
            targets = targets[valid_indices]

        if preds.shape[0] == 0:
            return

        # Calculate Sum of Squared Errors for this batch
        squared_diff = (preds - targets) ** 2
        batch_sse = np.sum(squared_diff, axis=0)
        batch_count = preds.shape[0]

        # Initialize storage if first update
        if self.sse_per_col is None:
            self.sse_per_col = np.zeros(preds.shape[1], dtype=np.float64)

        # Accumulate
        self.sse_per_col += batch_sse
        self.count += batch_count

    def compute(self):
        """
        Computes the final MCRMSE score based on accumulated statistics.

        Returns:
            float: The global MCRMSE score.
        """
        if self.count == 0:
            return 0.0

        # MSE per column
        mse_per_col = self.sse_per_col / self.count

        # RMSE per column
        rmse_per_col = np.sqrt(mse_per_col)

        # Mean across columns
        return np.mean(rmse_per_col)
