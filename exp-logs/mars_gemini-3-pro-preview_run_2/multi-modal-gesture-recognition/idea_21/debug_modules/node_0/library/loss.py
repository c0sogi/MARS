import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    CLASS_WEIGHTS,
    BOUNDARY_LOSS_WEIGHT,
    SMOOTHING_LOSS_WEIGHT,
    DEVICE,
)


class TMSELoss(nn.Module):
    """
    Temporal Mean Squared Error Loss (Smoothing Loss).
    Penalizes differences between probabilities at time t and t-1.
    """

    def __init__(self, threshold=None):
        super(TMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, probs, mask):
        """
        Args:
            probs: (B, T, C) Class probabilities.
            mask: (B, T) Sequence mask.
        Returns:
            loss: Scalar loss.
        """
        # Calculate difference between t and t-1
        # probs[:, 1:, :] corresponds to t
        # probs[:, :-1, :] corresponds to t-1
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Square the differences
        mse = diff.pow(2)

        # If a threshold is provided (Truncated MSE), clamp the values
        # However, instructions specify "do not clamp", so we skip or use high threshold if logic demanded.
        # We will implement standard MSE as per "do not clamp" instruction.
        if self.threshold is not None:
            mse = torch.clamp(mse, min=0, max=self.threshold)

        # Sum over classes (C) to get per-timestep error
        mse = torch.sum(mse, dim=2)  # (B, T-1)

        # Apply mask
        # The mask for diffs should be valid for both t and t-1.
        # mask[:, 1:] ensures t is valid. mask[:, :-1] ensures t-1 is valid.
        # Generally mask is 1 1 1 0 0.
        # t=1 (valid), t-1=0 (valid).
        valid_mask = mask[:, 1:] * mask[:, :-1]

        loss = torch.sum(mse * valid_mask)

        # Normalize by number of valid transitions
        num_valid = torch.sum(valid_mask)
        if num_valid > 0:
            loss = loss / num_valid

        return loss


class ActionSegmentationLoss(nn.Module):
    """
    Composite loss function for DSG-CRCN.
    Combines Classification, Boundary, and Smoothing losses with Deep Supervision.
    """

    def __init__(self):
        super(ActionSegmentationLoss, self).__init__()

        # Class Weights
        weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
        self.cls_criterion = nn.NLLLoss(weight=weights_tensor, reduction="none")

        # Boundary Criterion
        self.bnd_criterion = nn.BCELoss(reduction="none")

        # Smoothing Criterion
        self.smooth_criterion = TMSELoss(threshold=None)  # No clamping

        self.bnd_weight = BOUNDARY_LOSS_WEIGHT
        self.smooth_weight = SMOOTHING_LOSS_WEIGHT

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs: Dictionary containing outputs from all stages.
                Keys: 'stage1_cls', 'stage1_bnd', 'stage2_cls', ..., 'final_cls', ...
                Shapes: Cls (B, T, C), Bnd (B, T, 1)
            targets: Dictionary containing:
                'cls_labels': (B, T) Long
                'bnd_labels': (B, T) Float
                'mask': (B, T) Bool/Float
        """
        cls_targets = targets["cls_labels"]
        bnd_targets = targets["bnd_labels"]
        mask = targets["mask"].float()

        total_loss = 0.0
        metrics = {}

        # Iterate over stages
        stages = ["stage1", "stage2", "stage3"]

        for stage in stages:
            # Retrieve outputs
            # Model outputs are probabilities (Softmax/Sigmoid applied)
            cls_probs = model_outputs[f"{stage}_cls"]  # (B, T, C)
            bnd_probs = model_outputs[f"{stage}_bnd"]  # (B, T, 1)

            # 1. Classification Loss (Weighted Cross Entropy)
            # NLLLoss expects log-probabilities.
            # Add epsilon to prevent log(0)
            log_cls_probs = torch.log(cls_probs + 1e-7)

            # Flatten for NLLLoss: (B*T, C) vs (B*T)
            # But we need to apply mask. NLLLoss(reduction='none') returns (B, T)
            cls_loss_raw = self.cls_criterion(
                log_cls_probs.transpose(1, 2), cls_targets
            )  # Input: (B, C, T), Target: (B, T)
            cls_loss = torch.sum(cls_loss_raw * mask) / (torch.sum(mask) + 1e-7)

            # 2. Boundary Loss (Binary Cross Entropy)
            # bnd_probs: (B, T, 1) -> (B, T)
            bnd_probs_sq = bnd_probs.squeeze(2)
            bnd_loss_raw = self.bnd_criterion(bnd_probs_sq, bnd_targets)
            bnd_loss = torch.sum(bnd_loss_raw * mask) / (torch.sum(mask) + 1e-7)

            # 3. Smoothing Loss (TMSE)
            smooth_loss = self.smooth_criterion(cls_probs, mask)

            # Aggregate for this stage
            stage_loss = (
                cls_loss
                + (self.bnd_weight * bnd_loss)
                + (self.smooth_weight * smooth_loss)
            )
            total_loss += stage_loss

            # Record metrics for the final stage (usually stage3)
            if stage == "stage3":
                metrics["loss_cls"] = cls_loss.item()
                metrics["loss_bnd"] = bnd_loss.item()
                metrics["loss_smooth"] = smooth_loss.item()

        return total_loss, metrics
