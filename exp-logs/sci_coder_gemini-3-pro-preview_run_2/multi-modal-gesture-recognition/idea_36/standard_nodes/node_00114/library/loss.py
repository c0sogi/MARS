import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CLASS_WEIGHTS, LAMBDA_CLS, LAMBDA_BND, LAMBDA_SMOOTH


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for handling class imbalance in boundary detection.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: (B, T) or (B, T, 1) Logits
            targets: (B, T) Binary targets (0 or 1)
            mask: (B, T) Boolean mask
        """
        # Ensure inputs and targets have compatible shapes
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(-1)

        # Probabilities
        p = torch.sigmoid(inputs)

        # Binary Cross Entropy with Logits
        # Using functional for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Focal Term
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma

        loss = focal_factor * bce_loss

        # Alpha balancing
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if mask is not None:
            # Ensure mask is float for multiplication
            mask_float = mask.float()
            loss = loss * mask_float

            if self.reduction == "mean":
                # Normalize by number of valid elements
                return loss.sum() / (mask_float.sum() + 1e-6)
            elif self.reduction == "sum":
                return loss.sum()
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()

        return loss


class TMSELoss(nn.Module):
    """
    Temporal MSE Loss to enforce prediction smoothness.
    Computes MSE between probabilities at t and t-1.
    """

    def __init__(self, reduction="mean"):
        super(TMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, mask=None):
        """
        Args:
            logits: (B, T, C) Class logits
            mask: (B, T) Boolean mask
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Calculate diff: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # MSE over classes (dim=-1)
        # Shape: (B, T-1)
        loss_t = diff.pow(2).mean(dim=-1)

        if mask is not None:
            # We only care about transitions where BOTH t and t-1 are valid.
            # mask shape: (B, T)
            # mask_trans shape: (B, T-1)
            mask_trans = mask[:, 1:] & mask[:, :-1]
            mask_trans_float = mask_trans.float()

            loss_t = loss_t * mask_trans_float

            if self.reduction == "mean":
                return loss_t.sum() / (mask_trans_float.sum() + 1e-6)
            elif self.reduction == "sum":
                return loss_t.sum()
        else:
            if self.reduction == "mean":
                return loss_t.mean()
            elif self.reduction == "sum":
                return loss_t.sum()

        return loss_t


class DCHGLoss(nn.Module):
    """
    Aggregate loss function for Dense-Cascaded Hybrid-Gated Network.
    Combines Classification (CE), Boundary (Focal), and Smoothness (TMSE) losses
    across all 3 stages.
    """

    def __init__(self):
        super(DCHGLoss, self).__init__()

        # Classification Loss: Weighted Cross Entropy
        # We pass the weights from config. Note: CrossEntropyLoss copies the weights.
        self.cls_criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, reduction="none")

        # Boundary Loss: Focal Loss
        self.bnd_criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

        # Smoothness Loss: TMSE
        self.smooth_criterion = TMSELoss(reduction="mean")

        # Weights
        self.w_cls = LAMBDA_CLS
        self.w_bnd = LAMBDA_BND
        self.w_smooth = LAMBDA_SMOOTH

    def forward(self, outputs, targets, boundaries, mask):
        """
        Args:
            outputs: Dictionary containing model outputs:
                     'stage1_cls', 'stage1_bnd',
                     'stage2_cls', 'stage2_bnd',
                     'stage3_cls', 'stage3_bnd'
            targets: (B, T) Class indices (0-20)
            boundaries: (B, T) Binary boundary labels (0 or 1)
            mask: (B, T) Boolean mask indicating valid frames

        Returns:
            total_loss: Scalar tensor
            metrics: Dictionary of loss components for logging
        """
        total_loss = 0.0
        metrics = {}

        # Iterate over stages
        # The model is defined to have 3 stages
        stages = [1, 2, 3]

        for s in stages:
            # Retrieve logits
            cls_logits = outputs.get(f"stage{s}_cls")
            bnd_logits = outputs.get(f"stage{s}_bnd")

            if cls_logits is None or bnd_logits is None:
                continue

            # --- 1. Classification Loss ---
            # Reshape for CrossEntropy: (B*T, C) vs (B*T)
            B, T, C = cls_logits.shape

            # Compute raw element-wise loss
            cls_loss_raw = self.cls_criterion(
                cls_logits.reshape(-1, C), targets.view(-1)
            )
            cls_loss_raw = cls_loss_raw.view(B, T)

            # Apply mask and reduce
            mask_float = mask.float()
            cls_loss = (cls_loss_raw * mask_float).sum() / (mask_float.sum() + 1e-6)

            # --- 2. Boundary Loss ---
            # Focal loss handles masking internally if passed
            bnd_loss = self.bnd_criterion(bnd_logits, boundaries.float(), mask)

            # --- 3. Smoothness Loss ---
            smooth_loss = self.smooth_criterion(cls_logits, mask)

            # --- Aggregate ---
            stage_loss = (
                (self.w_cls * cls_loss)
                + (self.w_bnd * bnd_loss)
                + (self.w_smooth * smooth_loss)
            )

            total_loss += stage_loss

            # Log metrics
            metrics[f"loss_s{s}_cls"] = cls_loss.item()
            metrics[f"loss_s{s}_bnd"] = bnd_loss.item()
            metrics[f"loss_s{s}_smooth"] = smooth_loss.item()
            metrics[f"loss_s{s}_total"] = stage_loss.item()

        return total_loss, metrics
