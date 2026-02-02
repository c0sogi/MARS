import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from library.config import Config


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Computes the Gaussian radius for a given bounding box size such that
    the IoU with the ground truth box is at least min_overlap.
    Derived from CornerNet.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap at the specified center.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


class ModifiedFocalLoss(nn.Module):
    """
    Pixel-wise Modified Focal Loss for Heatmap Regression (CornerNet/CenterNet).
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        pred: (B, C, H, W) - Sigmoid output
        gt: (B, C, H, W) - Ground truth heatmap
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


class RegL1Loss(nn.Module):
    """
    L1 Loss masked by object presence.
    Used for Width/Height and Offset regression.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, target, mask):
        """
        output: (B, 2, H, W)
        target: (B, 2, H, W)
        mask: (B, H, W)
        """
        # Expand mask to match channel dim
        mask = mask.unsqueeze(1).expand_as(output).float()

        loss = F.l1_loss(output * mask, target * mask, reduction="sum")

        # Normalize by number of objects (sum of mask / 2 channels) + epsilon
        loss = loss / (mask.sum() / 2 + 1e-4)
        return loss


class CenterNetLoss(nn.Module):
    def __init__(self, hm_weight=1.0, wh_weight=0.1, off_weight=1.0, global_weight=1.0):
        super(CenterNetLoss, self).__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.global_weight = global_weight

        self.hm_loss = ModifiedFocalLoss()
        self.reg_loss = RegL1Loss()
        self.global_loss_fn = nn.BCELoss()

        # Downsampling ratio of the model (Stride 4)
        self.down_ratio = 4

    def forward(self, outputs, targets):
        """
        outputs: Dict containing 'hm', 'wh', 'reg', 'global_no_finding'
        targets: List of dicts (from collate_fn) or Dict of stacked tensors
        """
        hm_pred = outputs["hm"]
        wh_pred = outputs["wh"]
        reg_pred = outputs["reg"]
        global_pred = outputs["global_no_finding"]

        batch_size, num_classes, h, w = hm_pred.shape
        device = hm_pred.device

        # Initialize GT tensors
        hm_target = torch.zeros(
            (batch_size, num_classes, h, w), dtype=torch.float32, device=device
        )
        wh_target = torch.zeros(
            (batch_size, 2, h, w), dtype=torch.float32, device=device
        )
        reg_target = torch.zeros(
            (batch_size, 2, h, w), dtype=torch.float32, device=device
        )
        reg_mask = torch.zeros((batch_size, h, w), dtype=torch.float32, device=device)

        # Global head target: 1 if "No finding", 0 if finding
        # We collect this from the batch
        global_target_list = []

        # Process each image in the batch
        # Note: 'targets' comes from collate_fn as a tuple of dicts
        for b in range(batch_size):
            target_data = targets[b]

            # 1. Global Classification Target
            # target_data['cls_target'] is 1.0 if finding, 0.0 if no finding (based on data.py)
            # However, the model predicts "No Finding" probability.
            # So if finding (1.0), target should be 0.0. If no finding (0.0), target should be 1.0.
            is_finding = target_data["cls_target"].item()
            global_target_val = 1.0 - is_finding
            global_target_list.append(global_target_val)

            # If no objects, skip heatmap generation
            if "boxes" not in target_data or len(target_data["boxes"]) == 0:
                continue

            boxes = target_data["boxes"]  # Absolute coords (0 to IMG_SIZE)
            labels = target_data["labels"]

            for i in range(len(boxes)):
                bbox = boxes[i] / self.down_ratio
                cls_id = labels[i]

                # Bounding box coordinates on feature map
                x1, y1, x2, y2 = bbox

                # Width and Height on feature map
                bw = x2 - x1
                bh = y2 - y1

                if bw > 0 and bh > 0:
                    # Center
                    ct_x = (x1 + x2) / 2
                    ct_y = (y1 + y2) / 2

                    # Integral center
                    ct_x_int = int(ct_x)
                    ct_y_int = int(ct_y)

                    # Ensure within bounds
                    if (
                        ct_x_int >= 0
                        and ct_x_int < w
                        and ct_y_int >= 0
                        and ct_y_int < h
                    ):
                        # 1. Heatmap Target
                        radius = gaussian_radius((math.ceil(bh), math.ceil(bw)))
                        radius = max(0, int(radius))

                        # We need to draw on CPU numpy then convert back or implement pure torch
                        # Using numpy implementation for robustness as per standard CenterNet
                        hm_np = hm_target[b, cls_id].cpu().numpy()
                        draw_umich_gaussian(hm_np, (ct_x_int, ct_y_int), radius)
                        hm_target[b, cls_id] = torch.from_numpy(hm_np).to(device)

                        # 2. Width/Height Target
                        wh_target[b, 0, ct_y_int, ct_x_int] = bw
                        wh_target[b, 1, ct_y_int, ct_x_int] = bh

                        # 3. Offset Target (Local)
                        reg_target[b, 0, ct_y_int, ct_x_int] = ct_x - ct_x_int
                        reg_target[b, 1, ct_y_int, ct_x_int] = ct_y - ct_y_int

                        # 4. Mask
                        reg_mask[b, ct_y_int, ct_x_int] = 1

        # Stack global targets
        global_target = torch.tensor(
            global_target_list, dtype=torch.float32, device=device
        ).unsqueeze(1)

        # Compute Losses
        loss_hm = self.hm_loss(hm_pred, hm_target)
        loss_wh = self.reg_loss(wh_pred, wh_target, reg_mask)
        loss_off = self.reg_loss(reg_pred, reg_target, reg_mask)
        loss_global = self.global_loss_fn(global_pred, global_target)

        total_loss = (
            self.hm_weight * loss_hm
            + self.wh_weight * loss_wh
            + self.off_weight * loss_off
            + self.global_weight * loss_global
        )

        loss_stats = {
            "loss": total_loss,
            "hm_loss": loss_hm,
            "wh_loss": loss_wh,
            "off_loss": loss_off,
            "global_loss": loss_global,
        }

        return loss_stats
