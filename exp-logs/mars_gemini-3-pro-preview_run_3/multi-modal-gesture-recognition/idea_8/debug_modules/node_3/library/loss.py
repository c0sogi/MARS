import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add library path to access config
sys.path.append(os.path.abspath("./library"))
from config import Config


class BoundaryAdaptiveSmoothingLoss(nn.Module):
    """
    Computes the Boundary-Adaptive Smoothing Loss.

    This loss encourages the model to output temporally smooth predictions (constant class probabilities)
    within a gesture segment, but allows sharp transitions where the ground truth boundary indicates a change.

    Formula:
    L_smooth = Mean( (1 - y_bnd) * || log_p(t) - log_p(t-1) ||^2 )
    """

    def __init__(self):
        super(BoundaryAdaptiveSmoothingLoss, self).__init__()

    def forward(self, cls_logits, bnd_targets):
        """
        Args:
            cls_logits (torch.Tensor): Classification logits of shape (Batch, NumClasses, Time).
            bnd_targets (torch.Tensor): Boundary ground truth of shape (Batch, Time).
                                        Values should be 0.0 (no boundary) or 1.0 (boundary).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute Log Probabilities for numerical stability
        # (Batch, NumClasses, Time)
        log_probs = F.log_softmax(cls_logits, dim=1)

        # Calculate temporal difference: log_p(t) - log_p(t-1)
        # We slice the tensor to compute diffs between adjacent frames
        # diff[:, :, t] corresponds to log_probs[t+1] - log_probs[t]
        # Shape: (Batch, NumClasses, Time-1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Squared Euclidean Norm over the Class dimension (dim=1)
        # Shape: (Batch, Time-1)
        diff_sq = torch.sum(diff**2, dim=1)

        # Prepare Boundary Weights
        # We want to penalize smoothing only when there is NO boundary (y_bnd approx 0).
        # We align the weight with the first frame of the pair (t).
        # Shape: (Batch, Time-1)
        # Clamp targets to ensure stability, though inputs should be 0 or 1.
        bnd_weight = 1.0 - torch.clamp(bnd_targets[:, :-1], 0.0, 1.0)

        # Compute the weighted mean over the batch and time dimensions
        loss = torch.mean(bnd_weight * diff_sq)

        return loss


class MultiTaskCascadedLoss(nn.Module):
    """
    Aggregates losses from all stages of the BA-KC-IRN model.

    Components:
    1. Weighted Cross-Entropy Loss (Classification)
    2. Binary Cross-Entropy Loss (Boundary Detection)
    3. Boundary-Adaptive Smoothing Loss (Temporal Consistency)
    """

    def __init__(self):
        super(MultiTaskCascadedLoss, self).__init__()

        # 1. Setup Class Weights for CrossEntropy
        # Background class (0) is down-weighted to prevent it from dominating the loss
        weights = torch.ones(Config.NUM_CLASSES)
        weights[0] = Config.BG_WEIGHT
        # Register as buffer to automatically move to device with the module
        self.register_buffer("class_weights", weights)

        # 2. Initialize Loss Criteria
        self.cls_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.bnd_criterion = nn.BCEWithLogitsLoss()
        self.smooth_criterion = BoundaryAdaptiveSmoothingLoss()

        # 3. Load Hyperparameters
        self.lambda_cls = Config.LAMBDA_CLS
        self.lambda_bnd = Config.LAMBDA_BND
        self.lambda_smooth = Config.LAMBDA_SMOOTH

    def forward(self, outputs, cls_targets, bnd_targets):
        """
        Args:
            outputs (list): List of dictionaries from the model, one per stage.
                            Each dict contains:
                            - 'cls': (Batch, NumClasses, Time)
                            - 'bnd': (Batch, 1, Time)
            cls_targets (torch.Tensor): Ground truth labels (Batch, Time).
            bnd_targets (torch.Tensor): Ground truth boundaries (Batch, Time).

        Returns:
            torch.Tensor: Total aggregated loss.
        """
        total_loss = 0.0

        # Iterate over outputs from all stages (Deep Supervision)
        for stage_out in outputs:
            cls_logits = stage_out["cls"]  # (Batch, NumClasses, Time)
            bnd_logits = stage_out["bnd"]  # (Batch, 1, Time)

            # --- Classification Loss ---
            # CrossEntropyLoss expects (Batch, NumClasses, Time) vs (Batch, Time)
            loss_cls = self.cls_criterion(cls_logits, cls_targets)

            # --- Boundary Loss ---
            # BCEWithLogitsLoss expects (Batch, Time) vs (Batch, Time)
            # Squeeze the channel dimension of logits: (Batch, 1, Time) -> (Batch, Time)
            loss_bnd = self.bnd_criterion(bnd_logits.squeeze(1), bnd_targets)

            # --- Smoothing Loss ---
            loss_smooth = self.smooth_criterion(cls_logits, bnd_targets)

            # --- Aggregation ---
            stage_loss = (
                (self.lambda_cls * loss_cls)
                + (self.lambda_bnd * loss_bnd)
                + (self.lambda_smooth * loss_smooth)
            )

            total_loss += stage_loss

        return total_loss
