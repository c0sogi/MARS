import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class CascadedSmoothnessLoss(nn.Module):
    """
    Custom loss function for the Wide-Encoder Sawtooth Kinematic Network (WES-KN).

    Combines:
    1. Weighted Cross-Entropy Loss (Deep Supervision on all 3 stages).
    2. Log-Space Truncated MSE Loss (Temporal Smoothness on refinement stages).
    """

    def __init__(self):
        super(CascadedSmoothnessLoss, self).__init__()

        # Load hyperparameters from config
        # We clone the weights to ensure we have a dedicated tensor instance
        self.class_weights = config.CLASS_WEIGHTS.clone()
        self.smoothing_weight = config.SMOOTHING_LOSS_WEIGHT
        self.smoothing_threshold = config.SMOOTHING_THRESHOLD
        self.num_classes = config.NUM_CLASSES

    def truncated_mse_loss(self, log_probs):
        """
        Computes Truncated MSE over temporal differences in log-space.
        Loss = mean( clamp( (log_p_t - log_p_{t-1})^2, max=threshold^2 ) )

        Args:
            log_probs: Tensor of shape (Batch, Time, Classes)

        Returns:
            Scalar tensor representing the mean truncated MSE.
        """
        # Calculate temporal difference: P_t - P_{t-1}
        # Slice 1: to End, Slice 0: to End-1
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference
        mse = diff**2

        # Truncate gradients for large jumps (sharp boundaries)
        threshold_sq = self.smoothing_threshold**2
        truncated_mse = torch.clamp(mse, max=threshold_sq)

        return torch.mean(truncated_mse)

    def forward(self, outputs, targets):
        """
        Calculates the total loss.

        Args:
            outputs: Dictionary containing model outputs:
                     - 'logits_1': (Batch, Time, Classes)
                     - 'logits_2': (Batch, Time, Classes)
                     - 'logits_3': (Batch, Time, Classes)
            targets: Ground truth labels of shape (Batch, Time)

        Returns:
            Total loss (scalar).
        """
        device = targets.device
        weights = self.class_weights.to(device)

        # Flatten targets for CrossEntropy: (Batch * Time)
        targets_flat = targets.view(-1)

        # --- 1. Weighted Cross-Entropy (Deep Supervision) ---
        # We sum the CE loss from all three stages
        loss_ce = 0.0

        # Stage 1 (Encoder)
        l1 = outputs["logits_1"].reshape(-1, self.num_classes)
        loss_ce += F.cross_entropy(l1, targets_flat, weight=weights)

        # Stage 2 (Refinement 1)
        l2 = outputs["logits_2"].reshape(-1, self.num_classes)
        loss_ce += F.cross_entropy(l2, targets_flat, weight=weights)

        # Stage 3 (Refinement 2)
        l3 = outputs["logits_3"].reshape(-1, self.num_classes)
        loss_ce += F.cross_entropy(l3, targets_flat, weight=weights)

        # --- 2. Smoothing Loss (Truncated MSE) ---
        # Applied only to refinement stages to encourage smooth transitions
        # We use log_softmax for numerical stability in the difference calculation

        loss_smooth = 0.0

        # Stage 2 Smoothness
        log_probs_2 = F.log_softmax(outputs["logits_2"], dim=2)
        loss_smooth += self.truncated_mse_loss(log_probs_2)

        # Stage 3 Smoothness
        log_probs_3 = F.log_softmax(outputs["logits_3"], dim=2)
        loss_smooth += self.truncated_mse_loss(log_probs_3)

        # --- Total Loss ---
        total_loss = loss_ce + (self.smoothing_weight * loss_smooth)

        return total_loss
