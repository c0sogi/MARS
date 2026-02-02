import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) with strict masking.

    Logic:
    1. Slice data to the first 68 sequence positions (Config.PRED_LEN).
    2. Select only the scored columns (indices 0, 1, 3 corresponding to reactivity,
       deg_Mg_pH10, deg_Mg_50C).
    3. Compute MSE per column -> RMSE per column -> Mean over columns.
    """

    def __init__(self):
        super().__init__()
        # Indices corresponding to ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        # in the full list ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        self.scored_indices = torch.tensor([0, 1, 3], dtype=torch.long)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (N, 5, L)
            y_true: (N, 5, L)
        """
        # Ensure indices are on the correct device
        if self.scored_indices.device != y_pred.device:
            self.scored_indices = self.scored_indices.to(y_pred.device)

        # 1. Slice to valid sequence length (0 to 67)
        pred_scored = y_pred[:, :, : Config.PRED_LEN]
        true_scored = y_true[:, :, : Config.PRED_LEN]

        # 2. Select scored columns
        pred_scored = torch.index_select(pred_scored, 1, self.scored_indices)
        true_scored = torch.index_select(true_scored, 1, self.scored_indices)

        # 3. Compute MSE per column (averaging over Batch and Sequence dimensions)
        mse = torch.mean((pred_scored - true_scored) ** 2, dim=(0, 2))

        # 4. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Mean over columns
        return torch.mean(rmse)


class GlobalMCRMSE:
    """
    Accumulates squared errors and counts over the entire validation set
    to compute the global MCRMSE correctly.
    """

    def __init__(self):
        self.reset()
        # Indices corresponding to ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        self.scored_indices = torch.tensor([0, 1, 3], dtype=torch.long)

    def reset(self):
        self.sum_squared_errors = None  # Will be shape (3,)
        self.total_count = 0

    def update(self, y_pred, y_true):
        """
        Args:
            y_pred: (N, 5, L)
            y_true: (N, 5, L)
        """
        # Ensure indices are on the correct device
        if self.scored_indices.device != y_pred.device:
            self.scored_indices = self.scored_indices.to(y_pred.device)

        # 1. Slice to valid sequence length
        pred_scored = y_pred[:, :, : Config.PRED_LEN]
        true_scored = y_true[:, :, : Config.PRED_LEN]

        # 2. Select scored columns
        pred_scored = torch.index_select(pred_scored, 1, self.scored_indices)
        true_scored = torch.index_select(true_scored, 1, self.scored_indices)

        # 3. Calculate squared errors
        squared_errors = (pred_scored - true_scored) ** 2

        # Sum over batch (dim 0) and sequence (dim 2)
        # Result shape: (3,) corresponding to the 3 scored columns
        batch_sse = torch.sum(squared_errors, dim=(0, 2))

        # Count total elements per column (N * L)
        batch_count = pred_scored.shape[0] * pred_scored.shape[2]

        if self.sum_squared_errors is None:
            self.sum_squared_errors = torch.zeros_like(batch_sse)

        self.sum_squared_errors += batch_sse
        self.total_count += batch_count

    def compute(self):
        if self.total_count == 0:
            return 0.0

        # MSE per column
        mse = self.sum_squared_errors / self.total_count

        # RMSE per column
        rmse = torch.sqrt(mse)

        # Mean across columns
        score = torch.mean(rmse)

        return score.item()
