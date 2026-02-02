import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class WeightedCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with class weighting and sequence masking.
    Handles class imbalance by applying higher weights to gesture classes
    and lower weight to the background class.
    """

    def __init__(self, class_weights, device):
        super(WeightedCrossEntropy, self).__init__()
        # Convert list to tensor
        self.weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weights, reduction="none")

    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, T, C)
            targets: (B, T)
            mask: (B, T)
        """
        # Flatten for CrossEntropy
        B, T, C = logits.shape
        logits_flat = logits.view(-1, C)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        loss = self.criterion(logits_flat, targets_flat)

        # Apply mask
        loss = loss * mask_flat.float()

        # Normalize by number of valid tokens
        valid_tokens = torch.sum(mask_flat)
        if valid_tokens > 0:
            return torch.sum(loss) / valid_tokens
        else:
            return torch.sum(loss) * 0.0


class BoundaryBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss for boundary detection.
    Uses BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self):
        super(BoundaryBCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, T, 1)
            targets: (B, T) - Float tensor (0.0 or 1.0)
            mask: (B, T)
        """
        B, T, _ = logits.shape
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        loss = self.criterion(logits_flat, targets_flat)

        # Apply mask
        loss = loss * mask_flat.float()

        # Normalize
        valid_tokens = torch.sum(mask_flat)
        if valid_tokens > 0:
            return torch.sum(loss) / valid_tokens
        else:
            return torch.sum(loss) * 0.0


class TMSELoss(nn.Module):
    """
    Temporal Mean Squared Error (T-MSE) for smoothing.
    Penalizes rapid changes in class probabilities between adjacent frames.
    Per the prompt: Unclamped.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, probs, mask):
        """
        Args:
            probs: (B, T, C) - Softmax probabilities
            mask: (B, T)
        """
        # Calculate difference between t and t-1
        # P_t: probs[:, 1:, :]
        # P_{t-1}: probs[:, :-1, :]
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared Error
        loss = torch.pow(diff, 2)  # (B, T-1, C)

        # Sum over classes
        loss = torch.sum(loss, dim=2)  # (B, T-1)

        # Masking
        # Valid pairs are where both t and t-1 are valid.
        # Since mask is usually contiguous (1,1,1,0,0), we can use mask[:, 1:]
        valid_mask = mask[:, 1:].float()

        loss = loss * valid_mask

        # Normalize
        valid_pairs = torch.sum(valid_mask)
        if valid_pairs > 0:
            return torch.sum(loss) / valid_pairs
        else:
            return torch.sum(loss) * 0.0


class MultiStageLoss(nn.Module):
    """
    Aggregates losses from all stages of the CASGCN.
    """

    def __init__(self, device=config.DEVICE):
        super(MultiStageLoss, self).__init__()

        self.cls_loss_fn = WeightedCrossEntropy(config.CLASS_WEIGHTS, device)
        self.bnd_loss_fn = BoundaryBCELoss()
        self.smooth_loss_fn = TMSELoss()

        self.w_cls = config.LOSS_WEIGHT_CLS
        self.w_bnd = config.LOSS_WEIGHT_BND
        self.w_smooth = config.LOSS_WEIGHT_SMOOTH

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs: Dictionary with keys 'stage1', 'stage2', 'stage3'.
                           Each value is tuple (cls_logits, bnd_logits).
                           cls_logits: (B, T, C)
                           bnd_logits: (B, T, 1)
            targets: Dictionary with keys 'labels', 'boundaries', 'mask'.
        """
        labels = targets["labels"]
        boundaries = targets["boundaries"]
        mask = targets["mask"]

        total_loss = 0.0
        metrics = {}

        # Iterate over stages
        for stage_name, (cls_logits, bnd_logits) in model_outputs.items():

            # 1. Classification Loss
            l_cls = self.cls_loss_fn(cls_logits, labels, mask)

            # 2. Boundary Loss
            l_bnd = self.bnd_loss_fn(bnd_logits, boundaries, mask)

            # 3. Smoothing Loss (Only applies to probabilities)
            # Convert logits to probs
            probs = F.softmax(cls_logits, dim=2)
            l_smooth = self.smooth_loss_fn(probs, mask)

            # Weighted Sum
            stage_loss = (
                (self.w_cls * l_cls) + (self.w_bnd * l_bnd) + (self.w_smooth * l_smooth)
            )

            total_loss += stage_loss

            # Log metrics for this stage
            metrics[f"{stage_name}_loss"] = stage_loss.item()
            metrics[f"{stage_name}_cls"] = l_cls.item()
            metrics[f"{stage_name}_bnd"] = l_bnd.item()
            metrics[f"{stage_name}_smooth"] = l_smooth.item()

        return total_loss, metrics
