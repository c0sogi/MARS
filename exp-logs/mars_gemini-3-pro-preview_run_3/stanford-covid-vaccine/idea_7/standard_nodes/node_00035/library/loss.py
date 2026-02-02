import torch
import torch.nn as nn
from library.config import Config


class WeightedMCRMSELoss(nn.Module):
    """
    Weighted Mean Columnwise Root Mean Squared Error Loss.

    This loss function computes the RMSE for each target column separately and then
    averages them. It supports weighting individual predictions based on experimental
    error/uncertainty, allowing the model to focus on high-confidence data points
    (Noise-Aware Training).
    """

    def __init__(self):
        super(WeightedMCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED
        self.eps = 1e-8  # Epsilon for numerical stability

    def forward(self, preds, targets, weights=None):
        """
        Calculates the weighted MCRMSE loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Seq_Len, Targets).
                                  Example: (64, 107, 5)
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Scored, Targets).
                                    Example: (64, 68, 5)
            weights (torch.Tensor, optional): Weights for each prediction of shape (Batch, Seq_Scored, Targets).
                                              Example: (64, 68, 5).
                                              If None, standard MCRMSE is calculated.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Slice predictions to match the scored sequence length
        # The model outputs 107 positions, but we only have targets for the first 68.
        preds_sliced = preds[:, : self.seq_scored, :]

        # 2. Calculate Squared Errors
        squared_diff = (preds_sliced - targets) ** 2

        # 3. Compute MSE per column
        # We disable weighted loss as it hurts performance on the unweighted metric
        # (Cite solution_lesson_node_00034).
        if weights is not None and Config.USE_WEIGHTED_LOSS:
            # Apply weights: focus learning on high-confidence labels
            weighted_sq_diff = squared_diff * weights

            # Sum errors over batch (dim 0) and sequence (dim 1)
            sum_weighted_sq_diff = torch.sum(weighted_sq_diff, dim=(0, 1))

            # Sum weights over batch and sequence to normalize
            sum_weights = torch.sum(weights, dim=(0, 1))

            # Weighted MSE = Sum(w * error^2) / Sum(w)
            # Add epsilon to denominator to prevent division by zero
            mse_per_col = sum_weighted_sq_diff / (sum_weights + self.eps)
        else:
            # Standard MSE calculation (Unweighted)
            mse_per_col = torch.mean(squared_diff, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col + self.eps)

        # 5. Average RMSEs across all target columns (MCRMSE)
        loss = torch.mean(rmse_per_col)

        return loss
