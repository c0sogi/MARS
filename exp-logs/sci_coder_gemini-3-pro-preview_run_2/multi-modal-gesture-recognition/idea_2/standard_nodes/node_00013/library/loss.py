import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SmoothingLoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) loss for temporal smoothing.
    Penalizes rapid changes in log-probabilities between adjacent frames.
    """

    def __init__(self, threshold=4.0):
        """
        Args:
            threshold (float): The clamping threshold (tau) for the gradient of log-probabilities.
        """
        super(SmoothingLoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, log_probs, mask):
        """
        Args:
            log_probs: Log-probabilities of shape (Batch, Classes, Time)
            mask: Boolean mask of shape (Batch, Time) indicating valid frames.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate difference between t and t-1
        # shape: (Batch, Classes, Time-1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Clamp the difference to avoid exploding gradients and allow sharp transitions where necessary
        diff = torch.clamp(diff, min=-self.threshold, max=self.threshold)

        # Calculate Squared Error
        loss = self.mse(diff, torch.zeros_like(diff))

        # Apply mask
        # The mask for diffs should be the intersection of mask[t] and mask[t-1]
        # Generally, if mask is contiguous (1s then 0s), mask[:, 1:] is sufficient.
        mask_sliced = mask[:, 1:]  # (Batch, Time-1)

        # Expand mask to match classes dimension: (Batch, Classes, Time-1)
        mask_expanded = mask_sliced.unsqueeze(1).expand_as(loss)

        # Sum masked loss and divide by number of valid elements
        masked_loss = torch.sum(loss * mask_expanded)

        # Avoid division by zero
        num_valid = torch.sum(mask_expanded)
        if num_valid > 0:
            return masked_loss / num_valid
        else:
            return torch.tensor(0.0, device=log_probs.device, requires_grad=True)


class ActionSegmentationLoss(nn.Module):
    """
    Combined loss for MS-TCN: Weighted Cross Entropy + Smoothing Loss (T-MSE).
    Applied to the output of every stage.
    """

    def __init__(self, weight=None):
        """
        Args:
            weight (torch.Tensor, optional): Class weights for Cross Entropy.
                                             If None and Config.USE_WEIGHTED_LOSS is True,
                                             defaults to 1.0 for background and 3.0 for gestures.
        """
        super(ActionSegmentationLoss, self).__init__()

        self.lambda_smoothing = Config.LAMBDA_SMOOTHING

        # Initialize Class Weights
        if weight is None and Config.USE_WEIGHTED_LOSS:
            # Heuristic: Background (0) is dominant, so we weight gestures higher.
            # 1.0 for background, 3.0 for all other 20 classes.
            w = torch.ones(Config.NUM_CLASSES)
            w[1:] = 3.0
            self.ce_weight = w
        else:
            self.ce_weight = weight

        # We keep weights on CPU initially, move to device in forward if needed
        # or register as buffer to handle device movement automatically.
        if self.ce_weight is not None:
            self.register_buffer("class_weights", self.ce_weight)
        else:
            self.class_weights = None

        self.smoothing_loss = SmoothingLoss(threshold=4.0)

    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: List of tensors, one per stage. Each tensor is (Batch, Classes, Time).
            targets: Ground truth labels of shape (Batch, Time).
            mask: Boolean mask of shape (Batch, Time).

        Returns:
            total_loss: Aggregated loss across all stages.
        """
        total_loss = 0.0

        # Ensure mask is float for calculation if needed, though boolean is fine for indexing
        mask_float = mask.float()
        num_valid_frames = torch.sum(mask_float)

        for stage_out in predictions:
            # stage_out: (Batch, Classes, Time)

            # --- 1. Cross Entropy Loss ---
            # F.cross_entropy expects (Batch, Classes, Time) for multi-dimensional case
            ce_loss_per_frame = F.cross_entropy(
                stage_out, targets, weight=self.class_weights, reduction="none"
            )  # Output: (Batch, Time)

            # Apply mask
            masked_ce = torch.sum(ce_loss_per_frame * mask_float)
            if num_valid_frames > 0:
                stage_ce_loss = masked_ce / num_valid_frames
            else:
                stage_ce_loss = torch.tensor(0.0, device=stage_out.device)

            # --- 2. Smoothing Loss ---
            # Smoothing requires log_softmax probabilities
            log_probs = F.log_softmax(stage_out, dim=1)
            stage_smooth_loss = self.smoothing_loss(log_probs, mask)

            # --- 3. Aggregate ---
            total_loss += stage_ce_loss + (self.lambda_smoothing * stage_smooth_loss)

        return total_loss
