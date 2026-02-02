import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BoundaryAwareLoss(nn.Module):
    """
    Implements the Multi-Task Cascaded Loss for the BA-AKN model.
    Aggregates Classification, Boundary, and Gated Smoothing losses across all stages.
    """

    def __init__(self):
        super(BoundaryAwareLoss, self).__init__()

        # 1. Classification Loss
        # Weighted Cross Entropy to handle background vs gesture imbalance
        self.cls_criterion = nn.CrossEntropyLoss(weight=Config.CLASS_WEIGHTS)

        # 2. Boundary Loss
        # Binary Cross Entropy for transition detection
        # pos_weight could be added if boundaries are very sparse, but standard BCE is specified
        self.bnd_criterion = nn.BCEWithLogitsLoss()

        # Coefficients
        self.lambda_bnd = Config.LAMBDA_BND
        self.lambda_smooth = Config.LAMBDA_SMOOTH

    def gated_smoothing_loss(self, cls_logits, target_bnd):
        """
        Calculates the Adaptive Gated Smoothing Loss.
        L_adaptive = (1 - y_bnd) * || log P_t - log P_{t-1} ||^2

        Args:
            cls_logits (torch.Tensor): Shape (Batch, Classes, Time)
            target_bnd (torch.Tensor): Shape (Batch, Time). 1.0 at boundaries.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to log-probabilities
        # Shape: (Batch, Classes, Time)
        log_probs = F.log_softmax(cls_logits, dim=1)

        # Calculate temporal difference in log-space
        # diff[:, :, t] = log_probs[:, :, t+1] - log_probs[:, :, t]
        # Shape: (Batch, Classes, Time-1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Squared Euclidean distance (sum/mean over classes)
        # We take mean over classes to keep scale independent of num_classes
        # Shape: (Batch, Time-1)
        mse = torch.mean(diff**2, dim=1)

        # Prepare gating mask
        # target_bnd is 1.0 at boundaries, 0.0 inside gestures.
        # We want to disable smoothing at boundaries.
        # Gate = (1 - target_bnd).
        # We need to align the mask with the diffs.
        # We use the boundary label of the *next* frame (t+1) or current (t)?
        # If a transition occurs between t and t+1, we shouldn't smooth.
        # We'll use the max of boundary labels at t and t+1 to be safe,
        # or simply align with the diff indices.
        # Let's use the boundary values corresponding to the transition interval.
        # Shape: (Batch, Time-1)
        bnd_slice = target_bnd[:, 1:]
        gate = 1.0 - bnd_slice

        # Apply gate
        # If gate is 0 (boundary), loss is 0.
        # If gate is 1 (no boundary), loss is MSE.
        masked_mse = mse * gate

        return masked_mse.mean()

    def forward(self, outputs, targets_cls, targets_bnd):
        """
        Computes the total loss.

        Args:
            outputs (dict): Dictionary containing outputs from all stages.
                            Keys: 'stageX_cls', 'stageX_bnd'
            targets_cls (torch.Tensor): Ground truth class labels (Batch, Time).
            targets_bnd (torch.Tensor): Ground truth boundary labels (Batch, Time).

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        total_loss = 0.0
        metrics = {}

        # Iterate over stages defined in the model
        # We assume 3 stages based on Config and Model definition
        stages = [1, 2, 3]

        for s in stages:
            cls_key = f"stage{s}_cls"
            bnd_key = f"stage{s}_bnd"

            if cls_key not in outputs or bnd_key not in outputs:
                continue

            # Retrieve logits
            # cls_logits: (B, C, T)
            # bnd_logits: (B, 1, T)
            cls_logits = outputs[cls_key]
            bnd_logits = outputs[bnd_key]

            # --- 1. Classification Loss ---
            # CrossEntropyLoss expects (B, C, T) and Target (B, T)
            l_cls = self.cls_criterion(cls_logits, targets_cls)

            # --- 2. Boundary Loss ---
            # BCEWithLogitsLoss expects (B, 1, T) and Target (B, 1, T)
            # targets_bnd is (B, T), needs unsqueeze
            l_bnd = self.bnd_criterion(bnd_logits, targets_bnd.unsqueeze(1))

            # --- 3. Gated Smoothing Loss ---
            l_smooth = self.gated_smoothing_loss(cls_logits, targets_bnd)

            # Aggregate for this stage
            stage_loss = (
                l_cls + (self.lambda_bnd * l_bnd) + (self.lambda_smooth * l_smooth)
            )

            # Add to total
            total_loss += stage_loss

            # Record metrics for the final stage (or all, but usually final matters most for monitoring)
            metrics[f"loss_stage{s}"] = stage_loss.item()
            metrics[f"loss_cls_s{s}"] = l_cls.item()
            metrics[f"loss_bnd_s{s}"] = l_bnd.item()
            metrics[f"loss_smooth_s{s}"] = l_smooth.item()

        return total_loss, metrics
