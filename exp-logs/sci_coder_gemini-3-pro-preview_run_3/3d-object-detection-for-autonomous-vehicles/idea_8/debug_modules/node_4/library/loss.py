import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config
from library.utils import encode_refinement_targets


class GaussianFocalLoss(nn.Module):
    """
    Pixel-wise Focal Loss for Heatmap Regression.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - Sigmoid output
        target: (B, C, H, W) - Gaussian heatmap target [0, 1]
        """
        pos_inds = target.eq(1)
        neg_inds = target.lt(1)

        neg_weights = torch.pow(1 - target[neg_inds], self.beta)

        loss = 0

        # Positive loss
        # (1 - pred)^alpha * log(pred)
        pos_pred = pred[pos_inds]
        if pos_pred.numel() > 0:
            pos_loss = -torch.log(pos_pred + 1e-6) * torch.pow(1 - pos_pred, self.alpha)
            loss += pos_loss.sum()

        # Negative loss
        # (1 - target)^beta * pred^alpha * log(1 - pred)
        neg_pred = pred[neg_inds]
        if neg_pred.numel() > 0:
            neg_loss = (
                -torch.log(1 - neg_pred + 1e-6)
                * torch.pow(neg_pred, self.alpha)
                * neg_weights
            )
            loss += neg_loss.sum()

        num_pos = pos_inds.float().sum()
        if num_pos > 0:
            loss = loss / num_pos
        else:
            loss = (
                loss  # Avoid division by zero, though usually num_pos > 0 in training
            )

        return loss


class WeightedL1Loss(nn.Module):
    """
    L1 Loss applied only to masked regions (object centers).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        """
        pred: (B, C, H, W)
        target: (B, C, H, W)
        mask: (B, H, W) - 0 or 1
        """
        # Expand mask to match channels
        # mask: (B, 1, H, W)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask_expanded = mask.expand_as(pred)

        diff = torch.abs(pred - target) * mask_expanded

        # Normalize by number of positive pixels
        num_pos = mask.sum()
        if num_pos > 0:
            loss = diff.sum() / (num_pos + 1e-6)
        else:
            loss = diff.sum() * 0.0

        return loss


class RefinementLoss(nn.Module):
    """
    Loss for Stage 2. Handles matching proposals to GT and computing regression residuals.
    """

    def __init__(self, iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(self, residuals_pred, proposals, gt_boxes):
        """
        residuals_pred: (B, N, 8) - Predicted residuals
        proposals: (B, N, 7) - Stage 1 boxes
        gt_boxes: List of (M, 8) tensors - Ground truth boxes
        """
        batch_size = len(gt_boxes)
        total_loss = 0
        valid_batches = 0

        for b in range(batch_size):
            props = proposals[b]  # (N, 7)
            gts = gt_boxes[b]  # (M, 8)
            res_pred = residuals_pred[b]  # (N, 8)

            if len(gts) == 0 or len(props) == 0:
                continue

            # 1. Convert to AABB for fast IoU matching
            # props: x, y, z, w, l, h, yaw
            # AABB: x-w/2, y-l/2, x+w/2, y+l/2
            prop_aabb = torch.stack(
                [
                    props[:, 0] - props[:, 3] / 2,
                    props[:, 1] - props[:, 4] / 2,
                    props[:, 0] + props[:, 3] / 2,
                    props[:, 1] + props[:, 4] / 2,
                ],
                dim=1,
            )

            gt_aabb = torch.stack(
                [
                    gts[:, 0] - gts[:, 3] / 2,
                    gts[:, 1] - gts[:, 4] / 2,
                    gts[:, 0] + gts[:, 3] / 2,
                    gts[:, 1] + gts[:, 4] / 2,
                ],
                dim=1,
            )

            # 2. Calculate IoU
            # (N, M)
            iou = torchvision.ops.box_iou(prop_aabb, gt_aabb)

            # 3. Match
            # For each proposal, find best GT
            max_iou, gt_inds = iou.max(dim=1)

            # 4. Filter
            pos_mask = max_iou > self.iou_threshold

            if pos_mask.sum() == 0:
                continue

            pos_props = props[pos_mask]
            pos_gts = gts[gt_inds[pos_mask]]
            pos_res_pred = res_pred[pos_mask]

            # 5. Encode Targets
            # Use utility from library
            targets = encode_refinement_targets(pos_props, pos_gts)

            # 6. L1 Loss
            loss_b = F.l1_loss(pos_res_pred, targets)
            total_loss += loss_b
            valid_batches += 1

        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=proposals.device, requires_grad=True)


class TwoStageLoss(nn.Module):
    """
    Wrapper class to combine Stage 1 and Stage 2 losses.
    """

    def __init__(self):
        super().__init__()
        self.hm_loss_func = GaussianFocalLoss()
        self.reg_loss_func = WeightedL1Loss()
        self.refine_loss_func = RefinementLoss(iou_threshold=0.5)

        self.weight_hm = Config.LOSS_WEIGHT_HM
        self.weight_box = Config.LOSS_WEIGHT_BOX
        self.weight_refine = Config.LOSS_WEIGHT_REFINE

    def forward(self, stage1_maps, stage1_targets, stage2_preds, stage2_inputs):
        """
        Flexible forward method if losses were computed externally.

        stage1_maps: (hm_pred, reg_pred)
        stage1_targets: (hm_target, reg_target, reg_mask)
        stage2_preds: residuals_pred
        stage2_inputs: (proposals, gt_boxes)
        """
        hm_pred, reg_pred = stage1_maps
        hm_target, reg_target, reg_mask = stage1_targets

        # Stage 1
        loss_hm = self.hm_loss_func(hm_pred, hm_target) * self.weight_hm
        loss_box = self.reg_loss_func(reg_pred, reg_target, reg_mask) * self.weight_box

        # Stage 2
        residuals_pred = stage2_preds
        proposals, gt_boxes = stage2_inputs

        loss_refine = (
            self.refine_loss_func(residuals_pred, proposals, gt_boxes)
            * self.weight_refine
        )

        total_loss = loss_hm + loss_box + loss_refine

        return {
            "loss_hm": loss_hm,
            "loss_box": loss_box,
            "loss_refine": loss_refine,
            "total_loss": total_loss,
        }
