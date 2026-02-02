import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ActionSegmentationLoss(nn.Module):
    """
    Loss function for Multi-Stage Temporal Convolutional Networks (MS-TCN).
    Combines Cross-Entropy Loss for frame-wise classification and
    Truncated MSE for temporal smoothness.
    """

    def __init__(self, ignore_index: int = -100, mse_threshold: float = 4.0):
        """
        Args:
            ignore_index: Label index to ignore in CrossEntropyLoss (padding).
            mse_threshold: Threshold for truncating the Mean Squared Error in smoothing loss.
                           Standard value in MS-TCN literature is 4.0.
        """
        super(ActionSegmentationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mse_threshold = mse_threshold
        self.lambda_smooth = Config.LAMBDA_SMOOTH

    def compute_truncated_mse(
        self, predictions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Truncated Mean Squared Error between adjacent frame log-probabilities.
        This penalizes rapid fluctuations in predictions (jitter).

        Args:
            predictions: Logits tensor of shape (Batch, Classes, Time).
            mask: Boolean tensor of shape (Batch, Time) indicating valid frames.

        Returns:
            Scalar tensor representing the smoothing loss.
        """
        # Convert logits to log-probabilities
        log_probs = F.log_softmax(predictions, dim=1)

        # Calculate difference between frame t and t-1
        # Shape: (Batch, Classes, Time - 1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Squared Error
        loss_sq = diff**2

        # Truncate the error: min(diff^2, tau^2)
        loss_truncated = torch.clamp(loss_sq, min=0, max=self.mse_threshold**2)

        # Apply Masking to ignore padding
        if mask is not None:
            # We consider a transition valid if the current frame 't' is valid.
            # mask[:, 1:] corresponds to frames 1 to T-1.
            mask_sliced = mask[:, 1:]  # Shape: (Batch, Time - 1)

            # Expand mask to match feature dimensions: (Batch, Classes, Time - 1)
            mask_expanded = mask_sliced.unsqueeze(1).expand_as(loss_truncated)

            # Zero out loss for padded regions
            masked_loss = loss_truncated * mask_expanded.float()

            # Average over valid elements
            num_valid = mask_expanded.sum()
            if num_valid > 0:
                return masked_loss.sum() / num_valid
            else:
                return torch.tensor(0.0, device=predictions.device)
        else:
            return loss_truncated.mean()

    def forward(
        self, predictions: list, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the total multi-stage loss.

        Args:
            predictions: List of tensors, where each tensor is the output of a stage
                         with shape (Batch, Classes, Time).
            labels: Ground truth labels tensor of shape (Batch, Time).
            mask: Boolean mask tensor of shape (Batch, Time).

        Returns:
            Total loss scalar.
        """
        total_loss = 0.0

        for i, pred in enumerate(predictions):
            # 1. Cross Entropy Loss
            # Applied to all stages (Prediction + Refinement) to ensure every stage
            # learns to classify frames correctly.
            loss_ce = self.ce_loss(pred, labels)
            total_loss += loss_ce

            # 2. Smoothing Loss (Truncated MSE)
            # Applied only to Refinement Stages (Stage 2 onwards, i.e., index > 0).
            # This encourages the refinement network to correct jitter from the prediction network.
            if i > 0:
                loss_smooth = self.compute_truncated_mse(pred, mask)
                total_loss += self.lambda_smooth * loss_smooth

        return total_loss
