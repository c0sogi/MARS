import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import iou3d_global


def _neg_loss(pred, gt):
    """
    Modified focal loss. Exact same as CornerNet.
    Runs on a batch of heatmaps.
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class IoUAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = _neg_loss
        self.crit_reg = torch.nn.L1Loss(reduction="none")

        # Cache grid parameters from Config to avoid repeated lookups
        self.voxel_x = Config.VOXEL_SIZE[0] * Config.DOWN_RATIO
        self.voxel_y = Config.VOXEL_SIZE[1] * Config.DOWN_RATIO
        self.pc_range = Config.POINT_CLOUD_RANGE
        self.weights = Config.LOSS_WEIGHTS

    def forward(self, preds, targets):
        """
        Args:
            preds (dict): Output from the model containing 'hm', 'reg', 'wh', 'rot', 'z', 'iou'.
            targets (dict): Ground truth containing 'hm', 'ind', 'mask', 'reg', 'wh', 'rot', 'z'.
        """
        # 1. Heatmap Loss
        hm_loss = self.crit(preds["hm"], targets["hm"])

        target_mask = targets["mask"]
        ind = targets["ind"]

        # 2. Regression Losses
        # Helper to calculate reg loss for a specific head
        def get_reg_loss(name):
            if name not in preds:
                return torch.tensor(0.0, device=preds["hm"].device)
            pred = _transpose_and_gather_feat(preds[name], ind)
            target = targets[name]
            mask = target_mask.unsqueeze(2).expand_as(target).float()
            loss = self.crit_reg(pred, target) * mask
            loss = loss.sum() / (mask.sum() + 1e-4)
            return loss

        reg_loss = get_reg_loss("reg")
        wh_loss = get_reg_loss("wh")
        rot_loss = get_reg_loss("rot")
        z_loss = get_reg_loss("z")

        # 3. IoU Awareness Loss
        # We calculate the actual IoU between the predicted box (detached) and the GT box
        # and use this as the target for the 'iou' head.

        # Gather predictions at GT indices and detach gradients
        pred_reg = _transpose_and_gather_feat(preds["reg"], ind).detach()
        pred_wh = _transpose_and_gather_feat(preds["wh"], ind).detach()
        pred_rot = _transpose_and_gather_feat(preds["rot"], ind).detach()
        pred_z = _transpose_and_gather_feat(preds["z"], ind).detach()

        gt_reg = targets["reg"]
        gt_wh = targets["wh"]
        gt_rot = targets["rot"]
        gt_z = targets["z"]

        # Grid parameters for decoding
        B, K = ind.shape
        W = preds["hm"].shape[3]

        xs = (ind % W).float()
        ys = (ind // W).float()

        # Helper to reconstruct 3D boxes: (x, y, z, w, l, h, yaw)
        def get_boxes(reg, wh, z_v, rot):
            # x, y in metric coordinates
            bx = (xs + reg[..., 0]) * self.voxel_x + self.pc_range[0]
            by = (ys + reg[..., 1]) * self.voxel_y + self.pc_range[1]
            # dimensions
            bw = torch.exp(wh[..., 0])
            bl = torch.exp(wh[..., 1])
            bh = torch.exp(wh[..., 2])
            # height (z)
            bz = z_v[..., 0]
            # yaw
            brot = torch.atan2(rot[..., 0], rot[..., 1])
            return torch.stack([bx, by, bz, bw, bl, bh, brot], dim=-1)

        p_boxes_all = get_boxes(pred_reg, pred_wh, pred_z, pred_rot)
        g_boxes_all = get_boxes(gt_reg, gt_wh, gt_z, gt_rot)

        # Filter valid objects using the mask
        mask = target_mask.bool().view(-1)
        if mask.sum() > 0:
            p_boxes = p_boxes_all.view(-1, 7)[mask]
            g_boxes = g_boxes_all.view(-1, 7)[mask]

            # Calculate 3D IoU
            # iou3d_global returns (N, N) matrix, we need the diagonal (pairwise IoU)
            ious = iou3d_global(p_boxes, g_boxes)
            iou_targets = torch.diag(ious)

            # Clamp targets to valid range [0, 1]
            iou_targets = torch.clamp(iou_targets, min=0.0, max=1.0)

            # Get predicted IoU at the corresponding indices
            pred_iou = _transpose_and_gather_feat(preds["iou"], ind).view(-1, 1)
            pred_iou = pred_iou[mask.view(B, K)].squeeze()

            # IoU Loss (L1)
            iou_loss = F.l1_loss(pred_iou, iou_targets)
        else:
            iou_loss = torch.tensor(0.0, device=preds["hm"].device)

        # 4. Total Loss
        total_loss = (
            self.weights["hm"] * hm_loss
            + self.weights["reg"] * reg_loss
            + self.weights["wh"] * wh_loss
            + self.weights["rot"] * rot_loss
            + self.weights["z"] * z_loss
            + self.weights["iou"] * iou_loss
        )

        return total_loss, {
            "hm_loss": hm_loss.item(),
            "reg_loss": reg_loss.item(),
            "wh_loss": wh_loss.item(),
            "rot_loss": rot_loss.item(),
            "z_loss": z_loss.item(),
            "iou_loss": iou_loss.item(),
            "total_loss": total_loss.item(),
        }
