import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedCrossEntropy(nn.Module):
    """
    Weighted Cross Entropy Loss.
    Applies class weights defined in the Config to handle class imbalance,
    specifically down-weighting the background class.
    """

    def __init__(self):
        super(WeightedCrossEntropy, self).__init__()
        # Retrieve class weights from Config
        # We store them but ensure they are moved to the correct device in forward()
        self.weights = Config.get_class_weights()

    def forward(self, logits, targets):
        """
        Args:
            logits: (Batch, Time, Classes)
            targets: (Batch, Time)
        Returns:
            loss: Scalar tensor
        """
        # Ensure weights are on the same device as the input logits
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        # Reshape logits to (N, C) and targets to (N) for CrossEntropyLoss
        # N = Batch * Time
        num_classes = logits.shape[2]
        logits_flat = logits.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)

        return F.cross_entropy(logits_flat, targets_flat, weight=self.weights)


class LogSpaceSmoothingLoss(nn.Module):
    """
    Truncated Mean Squared Error (MSE) loss applied to log-probabilities.
    Enforces temporal continuity by penalizing rapid changes in prediction probabilities
    between adjacent frames. The truncation allows for valid sharp transitions
    at actual gesture boundaries.
    """

    def __init__(self, threshold=16.0):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits: (Batch, Time, Classes) - Raw logits from the model
        Returns:
            loss: Scalar tensor
        """
        # If sequence length is too short, smoothing loss is zero
        if logits.size(1) <= 1:
            return torch.tensor(0.0, device=logits.device)

        # Convert logits to log-probabilities: log(softmax(x))
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate difference between adjacent frames: P_t - P_{t-1}
        # Shape: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        mse = diff**2

        # Truncate the error to prevent over-penalizing valid boundaries
        # Standard practice in MS-TCN literature is to clamp at 16.0
        truncated_mse = torch.clamp(mse, min=0, max=self.threshold)

        # Return mean over all elements
        return torch.mean(truncated_mse)


class CascadedLoss(nn.Module):
    """
    Composite Loss Function for the Structurally-Augmented Attentive Kinematic Network.
    Aggregates losses from all three stages of the cascade.

    Formula:
    L_total = L_Stage1 + L_Stage2 + L_Stage3

    Where:
    L_Stage1 = WeightedCrossEntropy(P1, Y)
    L_Stage2 = WeightedCrossEntropy(P2, Y) + lambda * SmoothingLoss(P2)
    L_Stage3 = WeightedCrossEntropy(P3, Y) + lambda * SmoothingLoss(P3)
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()
        self.ce_loss = WeightedCrossEntropy()
        self.smooth_loss = LogSpaceSmoothingLoss()
        self.smooth_weight = Config.SMOOTHING_LOSS_WEIGHT

    def forward(self, logits1, logits2, logits3, targets):
        """
        Args:
            logits1: (Batch, Time, Classes) - Output from Stage 1 (BiGRU Encoder)
            logits2: (Batch, Time, Classes) - Output from Stage 2 (Refinement)
            logits3: (Batch, Time, Classes) - Output from Stage 3 (Refinement)
            targets: (Batch, Time) - Ground truth labels

        Returns:
            total_loss: Scalar tensor for backpropagation.
            metrics: Dictionary of individual loss components for logging.
        """
        # Stage 1: Sequence Encoder (Only Classification Loss)
        loss_ce1 = self.ce_loss(logits1, targets)

        # Stage 2: Refinement (Classification + Smoothing)
        loss_ce2 = self.ce_loss(logits2, targets)
        loss_smooth2 = self.smooth_loss(logits2)
        loss_stage2 = loss_ce2 + (self.smooth_weight * loss_smooth2)

        # Stage 3: Iterative Refinement (Classification + Smoothing)
        loss_ce3 = self.ce_loss(logits3, targets)
        loss_smooth3 = self.smooth_loss(logits3)
        loss_stage3 = loss_ce3 + (self.smooth_weight * loss_smooth3)

        # Total Loss
        total_loss = loss_ce1 + loss_stage2 + loss_stage3

        # Compile metrics
        metrics = {
            "loss_ce1": loss_ce1.item(),
            "loss_ce2": loss_ce2.item(),
            "loss_smooth2": loss_smooth2.item(),
            "loss_ce3": loss_ce3.item(),
            "loss_smooth3": loss_smooth3.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
