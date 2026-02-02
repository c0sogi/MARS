import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    CLASS_WEIGHTS,
    LOSS_LAMBDA_CLS,
    LOSS_LAMBDA_BND,
    LOSS_LAMBDA_SMOOTH,
)


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) for probability smoothing.
    According to the task description, this implementation does not clamp the loss,
    effectively acting as a temporal smoothness constraint on the probability distributions.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()

    def forward(self, probs, mask):
        """
        Args:
            probs: (N, L, C) Tensor of class probabilities (after Softmax).
            mask: (N, L) Tensor indicating valid frames.
        Returns:
            Scalar loss.
        """
        # Calculate difference between t and t-1
        # probs[:, 1:, :] corresponds to t
        # probs[:, :-1, :] corresponds to t-1
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared error
        mse = diff.pow(2).sum(dim=2)  # (N, L-1)

        # Apply mask
        # We use mask[:, 1:] because the first frame has no predecessor to compare with
        valid_mask = mask[:, 1:]

        # Avoid division by zero
        mask_sum = valid_mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        loss = (mse * valid_mask).sum() / mask_sum
        return loss


class DeepSupervisionLoss(nn.Module):
    """
    Composite loss function for the GSG-CRCN model.
    Aggregates Classification, Boundary Regression, and Smoothness losses
    across all three stages of the network.
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()

        # Initialize Class Weights for CrossEntropy
        # We register it as a buffer so it moves to the correct device with the module
        self.register_buffer(
            "class_weights", torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
        )

        self.tmse = TMSELoss()

        # Lambdas
        self.lambda_cls = LOSS_LAMBDA_CLS
        self.lambda_bnd = LOSS_LAMBDA_BND
        self.lambda_smooth = LOSS_LAMBDA_SMOOTH

    def compute_stage_loss(self, stage_out, cls_targets, bnd_targets, mask):
        """
        Computes the loss for a single stage.

        Args:
            stage_out: Dict containing 'cls' (N, L, C) and 'bnd' (N, L, 1) logits.
            cls_targets: (N, L) LongTensor.
            bnd_targets: (N, L) FloatTensor (Gaussian soft targets).
            mask: (N, L) Bool/Float Tensor.

        Returns:
            total_stage_loss, dict_of_components
        """
        cls_logits = stage_out["cls"]  # (N, L, C)
        bnd_logits = stage_out["bnd"]  # (N, L, 1)

        # Flatten for loss computation
        N, L, C = cls_logits.shape

        # --- 1. Classification Loss (Weighted Cross Entropy) ---
        # Reshape to (N*L, C)
        cls_logits_flat = cls_logits.reshape(-1, C)
        cls_targets_flat = cls_targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Compute element-wise CE loss
        ce_loss = F.cross_entropy(
            cls_logits_flat,
            cls_targets_flat,
            weight=self.class_weights,
            reduction="none",
        )

        # Apply mask
        mask_sum = mask_flat.sum()
        if mask_sum > 0:
            loss_cls = (ce_loss * mask_flat).sum() / mask_sum
        else:
            loss_cls = torch.tensor(0.0, device=cls_logits.device)

        # --- 2. Boundary Loss (MSE on Probabilities) ---
        # Apply Sigmoid to logits to get probabilities
        bnd_probs = torch.sigmoid(bnd_logits).squeeze(-1)  # (N, L)

        # MSE between predicted probability and Gaussian target
        bnd_mse = (bnd_probs - bnd_targets).pow(2)

        if mask_sum > 0:
            loss_bnd = (bnd_mse * mask).sum() / mask_sum
        else:
            loss_bnd = torch.tensor(0.0, device=cls_logits.device)

        # --- 3. Smoothness Loss (T-MSE on Class Probabilities) ---
        cls_probs = F.softmax(cls_logits, dim=2)
        loss_smooth = self.tmse(cls_probs, mask)

        # --- Total Stage Loss ---
        total_loss = (
            self.lambda_cls * loss_cls
            + self.lambda_bnd * loss_bnd
            + self.lambda_smooth * loss_smooth
        )

        return total_loss, {
            "loss_cls": loss_cls.item(),
            "loss_bnd": loss_bnd.item(),
            "loss_smooth": loss_smooth.item(),
        }

    def forward(self, model_output, cls_targets, bnd_targets, mask):
        """
        Args:
            model_output: Dictionary with keys 'stage1', 'stage2', 'stage3'.
            cls_targets: (N, L) Ground truth class indices.
            bnd_targets: (N, L) Ground truth boundary soft targets.
            mask: (N, L) Sequence mask.

        Returns:
            Total aggregated loss (Scalar).
            Dictionary containing breakdown of losses.
        """
        total_loss = 0.0
        metrics = {}

        stages = ["stage1", "stage2", "stage3"]

        for stage in stages:
            if stage in model_output:
                s_loss, s_metrics = self.compute_stage_loss(
                    model_output[stage], cls_targets, bnd_targets, mask
                )
                total_loss += s_loss

                # Store metrics prefixed with stage name
                for k, v in s_metrics.items():
                    metrics[f"{stage}_{k}"] = v

        return total_loss, metrics
