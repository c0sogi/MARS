import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from library.config import CFG


class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation.
    Tversky index is a generalization of Dice coefficient and Jaccard index.
    It adds weights (alpha, beta) to False Positives and False Negatives.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, probs, targets):
        """
        Args:
            probs: (B, C, H, W) Tensor, output of sigmoid (0-1)
            targets: (B, C, H, W) Tensor, binary ground truth (0 or 1)
        """
        # Flatten the tensors to (B*C, H*W) or just (N,)
        # We compute metric per class/sample or globally?
        # Usually per sample per class is best for stability, then mean.

        batch_size = probs.shape[0]
        num_classes = probs.shape[1]

        probs = probs.view(batch_size, num_classes, -1)
        targets = targets.view(batch_size, num_classes, -1)

        # True Positives, False Positives, False Negatives
        TP = (probs * targets).sum(dim=2)
        FP = ((1 - targets) * probs).sum(dim=2)
        FN = (targets * (1 - probs)).sum(dim=2)

        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1.0 - tversky_index.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss proposed by Kervadec et al.
    Minimizes the distance between predicted contours and GT contours.
    Uses Signed Distance Map (SDM) of the GT.
    """

    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def compute_sdf(self, mask):
        """
        Compute Signed Distance Function for a single binary mask using OpenCV.
        Args:
            mask: (H, W) numpy array, 0 (bg) or 1 (fg)
        Returns:
            sdf: (H, W) numpy array
                 Negative inside the object, Positive outside.
        """
        h, w = mask.shape
        # Check for empty mask
        if mask.sum() == 0:
            return None

        # Check for full mask (rare but possible)
        if mask.sum() == h * w:
            return -np.ones_like(mask, dtype=np.float32)

        mask_uint8 = mask.astype(np.uint8)

        # Distance to nearest foreground (0 inside, >0 outside)
        # We invert mask so FG is 0, BG is 1 for distanceTransform
        # dist_out: distance from background pixel to nearest foreground pixel
        dist_out = cv2.distanceTransform(1 - mask_uint8, cv2.DIST_L2, 5)

        # Distance to nearest background (0 outside, >0 inside)
        # mask: FG=1, BG=0. We want distance to 0.
        # dist_in: distance from foreground pixel to nearest background pixel
        dist_in = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

        # Signed Distance:
        # Inside: dist_out=0, dist_in>0 => sdf < 0
        # Outside: dist_out>0, dist_in=0 => sdf > 0
        sdf = dist_out - dist_in
        return sdf

    def forward(self, probs, targets):
        """
        Args:
            probs: (B, C, H, W) Tensor, output of sigmoid
            targets: (B, C, H, W) Tensor, binary ground truth
        """
        batch_size, num_classes, h, w = probs.shape

        # Convert targets to numpy for CPU processing
        targets_np = targets.detach().cpu().numpy()

        sdf_batch = np.zeros(probs.shape, dtype=np.float32)
        # Mask to track which samples have valid GT (non-empty)
        valid_mask_np = np.zeros((batch_size, num_classes, 1, 1), dtype=np.float32)

        # Compute SDFs on CPU
        for b in range(batch_size):
            for c in range(num_classes):
                mask = targets_np[b, c]
                sdf = self.compute_sdf(mask)
                if sdf is not None:
                    sdf_batch[b, c] = sdf
                    valid_mask_np[b, c] = 1.0
                else:
                    # Empty GT: Boundary loss undefined/skipped
                    pass

        # Convert back to tensor
        device = probs.device
        sdf_tensor = torch.from_numpy(sdf_batch).to(device)
        valid_mask = torch.from_numpy(valid_mask_np).to(device)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Loss = mean(probs * sdf) over valid pixels
        # We minimize this product.
        # Inside (sdf<0): maximize prob -> product becomes large negative
        # Outside (sdf>0): minimize prob -> product becomes 0

        weighted_probs = probs * sdf_tensor

        # Apply valid mask (broadcasting over H, W)
        loss = (weighted_probs * valid_mask).sum() / (valid_mask.sum() * h * w)

        return loss


class CompositeLoss(nn.Module):
    """
    Combines BCE, Tversky, and Boundary Loss.
    """

    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.boundary = BoundaryLoss()

        self.w_bce = CFG.bce_weight
        self.w_tversky = CFG.tversky_weight
        self.w_boundary = CFG.boundary_weight

        # Dictionary to store last computed metrics for logging
        self.last_metrics = {}

    def forward(self, logits, targets):
        # BCEWithLogitsLoss (Pixel-wise)
        bce_loss = self.bce(logits, targets)

        # Get probabilities for other losses
        probs = torch.sigmoid(logits)

        # Tversky Loss (Overlap / False Positive penalty)
        tversky_loss = self.tversky(probs, targets)

        # Boundary Loss (Shape / Distance)
        boundary_loss = self.boundary(probs, targets)

        # Weighted Sum
        total_loss = (
            (self.w_bce * bce_loss)
            + (self.w_tversky * tversky_loss)
            + (self.w_boundary * boundary_loss)
        )

        # Store metrics
        self.last_metrics = {
            "bce": bce_loss.item(),
            "tversky": tversky_loss.item(),
            "boundary": boundary_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss
