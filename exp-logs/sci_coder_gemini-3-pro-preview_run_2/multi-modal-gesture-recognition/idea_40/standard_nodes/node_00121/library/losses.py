import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import make_pad_mask


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for boundary detection.
    Focuses training on hard examples (sparse boundaries).
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (B, T, 1) or (B, T) Raw logits from the model.
            targets: (B, T) Binary targets (0 or 1).
            mask: (B, T) Boolean mask where True indicates padding (ignore).
        """
        # Ensure shapes match
        if logits.dim() > 2:
            logits = logits.squeeze(-1)

        # BCEWithLogitsLoss provides numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate pt
        pt = torch.exp(-bce_loss)

        # Focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Apply mask if provided
        if mask is not None:
            # mask is True for padding, so we invert it for valid positions
            valid_mask = ~mask
            focal_loss = focal_loss * valid_mask.float()

            if self.reduction == "mean":
                # Normalize by number of valid elements
                return focal_loss.sum() / (valid_mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return focal_loss.sum()
        else:
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()

        return focal_loss


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) for probability smoothing.
    As per instructions, this implementation does NOT clamp the loss (threshold=Infinity),
    effectively acting as a temporal smoothness constraint on Softmax probabilities.
    """

    def __init__(self, reduction="mean"):
        super(TMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, probs, mask=None):
        """
        Args:
            probs: (B, T, C) Softmax probabilities.
            mask: (B, T) Boolean mask where True indicates padding.
        """
        # Calculate temporal difference: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared difference
        mse = diff.pow(2).sum(dim=-1)  # Sum over classes -> (B, T-1)

        # Handle masking
        if mask is not None:
            # We need valid transitions. A transition is valid if both t and t-1 are valid.
            # mask is True for padding.
            # valid[t] = ~mask[t]
            # transition_valid[t-1] = valid[t-1] AND valid[t]
            valid_mask = ~mask
            transition_mask = valid_mask[:, :-1] & valid_mask[:, 1:]

            mse = mse * transition_mask.float()

            if self.reduction == "mean":
                return mse.sum() / (transition_mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return mse.sum()
        else:
            if self.reduction == "mean":
                return mse.mean()
            elif self.reduction == "sum":
                return mse.sum()

        return mse


class DeepSupervisionLoss(nn.Module):
    """
    Aggregated loss function for the GMG-CRGN model.
    Combines Weighted Cross-Entropy, Focal Loss (Boundary), and T-MSE (Smoothness)
    across multiple stages of the network.
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()

        # Class weights for handling imbalance (Background vs Gestures)
        self.class_weights = Config.get_class_weights()

        # Sub-losses
        # Note: CrossEntropyLoss expects logits
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        self.tmse_loss = TMSELoss(reduction="mean")

        # Weights for loss components
        self.lambda_cls = Config.LAMBDA_CLS
        self.lambda_bnd = Config.LAMBDA_BND
        self.lambda_smooth = Config.LAMBDA_SMOOTH

    def _generate_boundary_targets(self, targets):
        """
        Generates boundary targets from class labels.
        Boundary = 1 where label changes from previous frame, 0 otherwise.
        """
        # targets: (B, T)
        b_targets = torch.zeros_like(targets, dtype=torch.float32)

        # Compare t with t-1
        # 1 if different, 0 if same
        diff = (targets[:, 1:] != targets[:, :-1]).float()

        # Assign to t (change occurs at t)
        b_targets[:, 1:] = diff

        # First frame is assumed 0 (no change) unless we want to detect start,
        # but usually boundary is transition.

        return b_targets

    def forward(self, predictions, targets, lengths):
        """
        Args:
            predictions: List of tensors, one per stage.
                         Each tensor has shape (B, T, NumClasses + 1).
                         Last channel is boundary logit.
            targets: (B, T) Class indices.
            lengths: (B,) Sequence lengths.

        Returns:
            total_loss: Scalar tensor.
            metrics: Dictionary of individual loss components for logging.
        """
        device = targets.device

        # Ensure class weights are on the correct device
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
            self.ce_loss.weight = self.class_weights

        # Generate masks
        # make_pad_mask returns True for padding
        pad_mask = make_pad_mask(lengths, max_len=targets.size(1))
        valid_mask = ~pad_mask

        # Generate boundary targets
        boundary_targets = self._generate_boundary_targets(targets)

        total_loss = 0.0
        metrics = {}

        num_stages = len(predictions)

        for i, stage_pred in enumerate(predictions):
            # stage_pred: (B, T, C+1)

            # Split into Classification (Logits) and Boundary (Logits)
            cls_logits = stage_pred[:, :, : Config.NUM_CLASSES]  # (B, T, 21)
            bnd_logits = stage_pred[:, :, Config.NUM_CLASSES]  # (B, T)

            # --- 1. Classification Loss (Weighted CE) ---
            # Flatten for CE Loss: (B*T, C) and (B*T)
            # We use reduction='none' and apply mask manually to handle padding correctly
            loss_cls_raw = self.ce_loss(cls_logits.transpose(1, 2), targets)  # (B, T)
            loss_cls = (loss_cls_raw * valid_mask.float()).sum() / (
                valid_mask.sum() + 1e-8
            )

            # --- 2. Boundary Loss (Focal) ---
            loss_bnd = self.focal_loss(bnd_logits, boundary_targets, mask=pad_mask)

            # --- 3. Smoothness Loss (T-MSE) ---
            # Apply Softmax to get probabilities for T-MSE
            cls_probs = F.softmax(cls_logits, dim=-1)
            loss_smooth = self.tmse_loss(cls_probs, mask=pad_mask)

            # Aggregate for this stage
            stage_loss = (
                self.lambda_cls * loss_cls
                + self.lambda_bnd * loss_bnd
                + self.lambda_smooth * loss_smooth
            )

            total_loss += stage_loss

            # Logging
            metrics[f"loss_stage_{i+1}"] = stage_loss.item()
            metrics[f"loss_cls_{i+1}"] = loss_cls.item()
            metrics[f"loss_bnd_{i+1}"] = loss_bnd.item()
            metrics[f"loss_smooth_{i+1}"] = loss_smooth.item()

        return total_loss, metrics
