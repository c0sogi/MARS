import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for RNA degradation prediction.

    Handles:
    1. Masking/Slicing to the scored sequence length (default 68).
    2. Selecting specific scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Weighted aggregation of losses from multiple recycling iterations (if input is a list).
    4. Automatic shape alignment between predictions (B, C, L) and targets (B, L, C).
    """

    def __init__(
        self,
        scored_length=Config.SCORED_LENGTH,
        scored_indices=Config.SCORED_COLS_INDICES,
        weights=None,
    ):
        super().__init__()
        self.scored_length = scored_length
        self.scored_indices = scored_indices
        # Default weights correspond to [pass_1_weight, pass_2_weight] defined in Idea 41
        self.weights = weights if weights is not None else [0.5, 1.0]

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (B, C, L) or (B, L, C), or a list of such Tensors.
            targets: Tensor of shape (B, L, C).
        Returns:
            Scalar loss value.
        """
        # Handle list input (Recycling/Iterative refinement)
        if isinstance(preds, list):
            total_loss = 0
            # Adjust weights if length mismatches (fallback safety)
            current_weights = self.weights
            if len(preds) != len(current_weights):
                # If mismatch, default to equal weighting or last-only?
                # We'll default to 1.0 for all to avoid silent failure,
                # but typically this should match Config.N_CYCLES.
                current_weights = [1.0] * len(preds)

            for i, pred in enumerate(preds):
                loss = self._compute_single_loss(pred, targets)
                total_loss += current_weights[i] * loss
            return total_loss

        # Handle single tensor input
        else:
            return self._compute_single_loss(preds, targets)

    def _compute_single_loss(self, pred, target):
        """
        Computes MCRMSE for a single prediction tensor.
        """
        # 1. Align Shapes
        # Target is typically (B, L, C). Model output might be (B, C, L).
        if pred.shape != target.shape:
            # Check if transposing the last two dims fixes the shape match
            if pred.shape[1] == target.shape[2] and pred.shape[2] == target.shape[1]:
                pred = pred.transpose(1, 2)

        # 2. Slice to Scored Length (First 68 positions)
        # Assuming shape is now (B, L, C)
        pred_scored = pred[:, : self.scored_length, :]
        target_scored = target[:, : self.scored_length, :]

        # 3. Select Scored Columns
        # Indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
        pred_scored = pred_scored[:, :, self.scored_indices]
        target_scored = target_scored[:, :, self.scored_indices]

        # 4. Compute MSE
        # Mean over Batch (0) and Sequence Length (1)
        mse = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 1))

        # 5. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 6. Average RMSE across columns (MCRMSE)
        return torch.mean(rmse)
