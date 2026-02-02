import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class TruncatedMSELoss(nn.Module):
    """
    Computes a Truncated Mean Squared Error loss on the temporal differences
    of log-probabilities. This encourages smoothness in predictions while
    allowing for sharp transitions (boundaries) by capping the penalty.
    """

    def __init__(self, threshold):
        """
        Args:
            threshold (float): The maximum value for the squared difference error.
                               Errors larger than this are clamped.
        """
        super(TruncatedMSELoss, self).__init__()
        self.threshold_sq = threshold**2

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Raw model outputs of shape (Batch, Class, Time).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to log-probabilities for numerical stability
        # Shape: (B, C, T)
        log_probs = F.log_softmax(logits, dim=1)

        # Calculate temporal difference: log_P(t) - log_P(t-1)
        # Slicing creates diff of shape (B, C, T-1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Squared Error
        squared_error = diff.pow(2)

        # Truncate the error to prevent over-penalizing valid sharp transitions
        truncated_error = torch.clamp(squared_error, max=self.threshold_sq)

        # Return the mean over the batch, classes, and time
        return truncated_error.mean()


class CascadedLoss(nn.Module):
    """
    Computes the total loss for the multi-stage architecture (Deep Supervision).
    Combines Weighted Cross-Entropy for classification and Truncated MSE for
    temporal smoothness on refinement stages.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Define Class Weights
        # Initialize with 1.0 for all classes
        weights = torch.ones(config.NUM_CLASSES)
        # Apply specific weight for the background class (Class 0)
        weights[config.BACKGROUND_CLASS_ID] = config.BACKGROUND_WEIGHT

        # Register as a buffer so it moves to the correct device (CPU/GPU) automatically
        self.register_buffer("class_weights", weights)

        # Weighted Cross Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

        # Smoothing Loss
        self.smooth_loss = TruncatedMSELoss(threshold=config.SMOOTHING_THRESHOLD)
        self.smooth_weight = config.SMOOTHING_LOSS_WEIGHT

    def forward(self, predictions, targets):
        """
        Args:
            predictions (list of torch.Tensor): List containing outputs from each stage.
                                                [Stage1_Logits, Stage2_Logits, Stage3_Logits]
                                                Each tensor has shape (B, C, T).
            targets (torch.Tensor): Ground truth labels of shape (B, T).

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        total_loss = 0.0
        metrics = {}

        # --- Stage 1: High-Capacity Kinematic Sequence Encoder ---
        # Only Classification Loss
        p1 = predictions[0]
        loss_1_ce = self.ce_loss(p1, targets)

        total_loss += loss_1_ce
        metrics["loss_s1_ce"] = loss_1_ce.item()

        # --- Stage 2: Hierarchical Sawtooth Refinement ---
        # Classification Loss + Smoothing Loss
        if len(predictions) > 1:
            p2 = predictions[1]
            loss_2_ce = self.ce_loss(p2, targets)
            loss_2_smooth = self.smooth_loss(p2)

            total_loss += loss_2_ce + (self.smooth_weight * loss_2_smooth)

            metrics["loss_s2_ce"] = loss_2_ce.item()
            metrics["loss_s2_smooth"] = loss_2_smooth.item()

        # --- Stage 3: Independent Iterative Refinement ---
        # Classification Loss + Smoothing Loss
        if len(predictions) > 2:
            p3 = predictions[2]
            loss_3_ce = self.ce_loss(p3, targets)
            loss_3_smooth = self.smooth_loss(p3)

            total_loss += loss_3_ce + (self.smooth_weight * loss_3_smooth)

            metrics["loss_s3_ce"] = loss_3_ce.item()
            metrics["loss_s3_smooth"] = loss_3_smooth.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
