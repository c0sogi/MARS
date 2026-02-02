import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from library.config import Config


class BCETverskyLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy and Tversky Loss.
    Used as the primary objective during the warmup phase.
    """

    def __init__(self, alpha=0.5, beta=0.5, bce_weight=0.5, smooth=1.0):
        """
        Args:
            alpha (float): Weight for False Positives in Tversky (default 0.5 for Dice).
            beta (float): Weight for False Negatives in Tversky (default 0.5 for Dice).
            bce_weight (float): Weight assigned to BCE loss (0.0 to 1.0).
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(BCETverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, preds, targets):
        # preds: Logits (B, C, H, W)
        # targets: Binary mask (B, C, H, W)

        # Binary Cross Entropy
        bce_loss = self.bce(preds, targets)

        # Tversky Loss
        preds_prob = torch.sigmoid(preds)

        # Flatten tensors for Tversky calculation
        preds_flat = preds_prob.view(preds_prob.size(0), preds_prob.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        # True Positives, False Positives, False Negatives
        tp = (preds_flat * targets_flat).sum(dim=2)
        fp = (preds_flat * (1 - targets_flat)).sum(dim=2)
        fn = ((1 - preds_flat) * targets_flat).sum(dim=2)

        tversky_score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        tversky_loss = 1 - tversky_score.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * tversky_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss that penalizes the distance between predicted probabilities and
    the ground truth boundary. Uses Signed Distance Maps (SDM).
    Used in the refinement phase to optimize Hausdorff distance.
    """

    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def compute_sdf(self, mask_np):
        """
        Computes the normalized Signed Distance Map for a single binary mask.
        """
        # Handle empty mask case
        if mask_np.sum() == 0:
            # If ground truth is empty, any positive prediction is a large error.
            # We return a positive distance map (e.g., all ones) to penalize any activation.
            return np.ones_like(mask_np)

        posmask = mask_np.astype(bool)
        negmask = ~posmask

        # Handle full mask case (rare but possible)
        if posmask.all():
            return -np.ones_like(mask_np)

        # Distance to nearest background (0)
        d_in = distance_transform_edt(posmask)
        # Distance to nearest foreground (1)
        d_out = distance_transform_edt(negmask)

        # Signed Distance: Negative inside, Positive outside
        sdf = d_out - d_in

        # Normalize by image height to keep loss magnitude stable across resolutions
        scale = mask_np.shape[0]
        return sdf / scale

    def forward(self, preds, targets):
        # preds: Logits (B, C, H, W)
        # targets: Binary mask (B, C, H, W)

        preds_prob = torch.sigmoid(preds)

        # Compute SDF for targets on CPU (scipy requirement)
        # Detach targets to ensure no gradient flows back into target generation (not needed anyway)
        targets_np = targets.detach().cpu().numpy()

        sdf_batch = []
        for b in range(targets_np.shape[0]):
            sdf_channels = []
            for c in range(targets_np.shape[1]):
                sdf = self.compute_sdf(targets_np[b, c])
                sdf_channels.append(sdf)
            sdf_batch.append(np.stack(sdf_channels))

        # Stack and convert back to tensor
        sdf_batch = np.stack(sdf_batch)  # (B, C, H, W)
        sdf_tensor = torch.from_numpy(sdf_batch).to(preds.device, dtype=preds.dtype)

        # Loss is the mean product of Probability and Signed Distance
        # P=1, Dist>0 (Outside) -> Positive Loss (Penalty)
        # P=1, Dist<0 (Inside)  -> Negative Loss (Reward)
        # P=0, Dist>0 (Outside) -> Zero Loss (Correct)
        loss = (preds_prob * sdf_tensor).mean()
        return loss


class CurriculumLoss(nn.Module):
    """
    Wrapper class that handles Deep Supervision.
    Simplified to use only BCE + Tversky Loss, removing Boundary Loss to prioritize
    spatial resolution and training speed (Cite Lesson 00027).
    """

    def __init__(self, config):
        super(CurriculumLoss, self).__init__()
        self.config = config
        # Loss: BCE + Tversky (Dice)
        self.bce_tversky = BCETverskyLoss(alpha=0.5, beta=0.5, bce_weight=0.5)

    def forward(self, preds, targets, epoch):
        """
        Args:
            preds: Model output. Can be a tensor (B, C, H, W) or a list of tensors (Deep Supervision).
            targets: Ground truth mask (B, C, H, W).
            epoch: Current training epoch (0-indexed).
        """
        # Handle Deep Supervision (List of outputs)
        if isinstance(preds, (list, tuple)):
            loss = 0
            # We average the loss across all decoder levels
            for p in preds:
                loss += self._compute_single_loss(p, targets)
            return loss / len(preds)
        else:
            return self._compute_single_loss(preds, targets)

    def _compute_single_loss(self, pred, target):
        # Resize target if prediction shape doesn't match (e.g., lower resolution decoder outputs)
        if pred.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=pred.shape[2:], mode="nearest")

        # Base Loss (Always active)
        return self.bce_tversky(pred, target)
