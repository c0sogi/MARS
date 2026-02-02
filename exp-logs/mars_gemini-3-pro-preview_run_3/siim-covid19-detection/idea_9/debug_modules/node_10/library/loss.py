import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU/GIoU between two sets of bounding boxes.
    box1: [N, 4]
    box2: [M, 4]
    Returns: [N, M] matrix of IoUs
    """
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Intersection area
    inter = (torch.min(b1_x2[:, None], b2_x2) - torch.max(b1_x1[:, None], b2_x1)).clamp(
        0
    ) * (torch.min(b1_y2[:, None], b2_y2) - torch.max(b1_y1[:, None], b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1[:, None] * h1[:, None] + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2[:, None], b2_x2) - torch.min(
            b1_x1[:, None], b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2[:, None], b2_y2) - torch.min(
            b1_y1[:, None], b2_y1
        )  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1[:, None] - b1_x2[:, None]) ** 2
                + (b2_y1 + b2_y2 - b1_y1[:, None] - b1_y2[:, None]) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / (3.1415926535**2)) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1[:, None] / h1[:, None]), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area

    return iou


class QualityFocalLoss(nn.Module):
    """
    Generalized Focal Loss (Quality Focal Loss) for classification.
    """

    def __init__(self, beta=2.0, reduction="mean"):
        super(QualityFocalLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred_logits, target_quality):
        """
        pred_logits: [N, C] (before sigmoid)
        target_quality: [N, C] (0 for neg, IoU for pos)
        """
        pred_sigmoid = pred_logits.sigmoid()
        scale_factor = pred_sigmoid - target_quality
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_quality, reduction="none"
        ) * scale_factor.abs().pow(self.beta)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


class GIoULoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(GIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes: [N, 4] (x1, y1, x2, y2)
        target_boxes: [N, 4] (x1, y1, x2, y2)
        """
        # Calculate GIoU. Since inputs are matched 1-to-1, we use diagonal of the matrix or simplified calc
        # Re-using the matrix function but taking diagonal is inefficient, implementing element-wise

        b1_x1, b1_y1, b1_x2, b1_y2 = (
            pred_boxes[:, 0],
            pred_boxes[:, 1],
            pred_boxes[:, 2],
            pred_boxes[:, 3],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            target_boxes[:, 0],
            target_boxes[:, 1],
            target_boxes[:, 2],
            target_boxes[:, 3],
        )

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = area1 + area2 - inter_area + 1e-7

        iou = inter_area / union_area

        # Enclosing box
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)

        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-7
        giou = iou - (c_area - union_area) / c_area

        loss = 1.0 - giou

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss


class ATSSMatcher(nn.Module):
    """
    Adaptive Training Sample Selection (ATSS) Matcher.
    Assigns targets to anchors based on statistical characteristics of IoU.
    """

    def __init__(self, topk=9):
        super(ATSSMatcher, self).__init__()
        self.topk = topk

    def forward(self, anchors, gt_boxes, num_anchors_per_level):
        """
        anchors: [N_anchors, 4] (cx, cy, stride_w, stride_h)
        gt_boxes: [N_gt, 4] (x1, y1, x2, y2)
        num_anchors_per_level: list of int, number of anchors in each FPN level

        Returns:
            matched_gt_inds: [N_anchors] -1 for ignore, 0+ for gt index, -2 for background
            anchor_iou_gt: [N_anchors] IoU with assigned GT
        """
        num_anchors = anchors.size(0)
        num_gt = gt_boxes.size(0)

        if num_gt == 0:
            return torch.full(
                (num_anchors,), -2, dtype=torch.long, device=anchors.device
            ), torch.zeros((num_anchors,), dtype=torch.float32, device=anchors.device)

        # 1. Calculate IoU between all anchors and all GTs
        # Convert anchors to x1y1x2y2 for IoU calc
        # Anchors are cx, cy, stride_w, stride_h. We approximate box size as stride * scalar or use explicit scales
        # Here we assume anchors represent the center points and strides.
        # ATSS usually uses a base scale (e.g. 8*stride).
        # However, for simplicity and robustness, we assume anchors input here are already expanded to [x1,y1,x2,y2]
        # or we construct them.
        # Let's assume input `anchors` are [cx, cy, stride_w, stride_h].
        # We construct probe boxes for IoU calc: size = stride * 8 (standard ATSS config)

        anchor_cx = anchors[:, 0]
        anchor_cy = anchors[:, 1]
        # Use stride as proxy for scale if explicit width/height not provided
        anchor_w = anchors[:, 2] * 8.0
        anchor_h = anchors[:, 3] * 8.0

        anchor_boxes_x1y1x2y2 = torch.stack(
            [
                anchor_cx - anchor_w / 2,
                anchor_cy - anchor_h / 2,
                anchor_cx + anchor_w / 2,
                anchor_cy + anchor_h / 2,
            ],
            dim=1,
        )

        ious = bbox_iou(anchor_boxes_x1y1x2y2, gt_boxes)  # [N_anchors, N_gt]

        # 2. Calculate distances between anchor centers and GT centers
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

        # [N_anchors, N_gt]
        distances = (anchor_cx[:, None] - gt_cx[None, :]).pow(2) + (
            anchor_cy[:, None] - gt_cy[None, :]
        ).pow(2)
        distances = distances.sqrt()

        # 3. Select candidates
        candidate_idxs = []
        start_idx = 0
        for n_anchors_level in num_anchors_per_level:
            end_idx = start_idx + n_anchors_level

            # For each GT, select topk anchors with smallest center distance in this level
            level_distances = distances[start_idx:end_idx, :]  # [N_level, N_gt]

            k = min(self.topk, n_anchors_level)
            _, topk_idxs = level_distances.topk(k, dim=0, largest=False)  # [k, N_gt]

            candidate_idxs.append(topk_idxs + start_idx)
            start_idx = end_idx

        candidate_idxs = torch.cat(candidate_idxs, dim=0)  # [K*L, N_gt]

        # 4. Compute Thresholds
        # Gather IoUs for candidates
        # candidate_ious: [K*L, N_gt]
        candidate_ious = ious.gather(0, candidate_idxs)

        iou_mean = candidate_ious.mean(dim=0)
        iou_std = candidate_ious.std(dim=0)
        iou_thresh = iou_mean + iou_std

        # 5. Assign Positives
        is_pos = torch.zeros_like(ious, dtype=torch.bool)

        # Condition 1: Candidate IoU > Threshold
        # We need to scatter back to full matrix
        # Create a mask for candidates
        candidate_mask = torch.zeros_like(ious, dtype=torch.bool)
        candidate_mask.scatter_(0, candidate_idxs, True)

        is_pos = candidate_mask & (ious >= iou_thresh[None, :])

        # Condition 2: Center in GT
        # anchor centers: [N_anchors, 2]
        # gt boxes: [N_gt, 4]
        # Check if anchor center is inside gt box
        # l < cx < r  AND  t < cy < b
        lt = anchors[:, :2][:, None, :] - gt_boxes[None, :, :2]  # [N_a, N_gt, 2]
        rb = gt_boxes[None, :, 2:] - anchors[:, :2][:, None, :]  # [N_a, N_gt, 2]
        is_in_gt = (torch.cat([lt, rb], dim=2) > 0.01).all(dim=2)  # [N_a, N_gt]

        is_pos = is_pos & is_in_gt

        # 6. Handle Ambiguities (One anchor assigned to multiple GTs)
        # If an anchor is positive for multiple GTs, assign to the one with max IoU
        # But first, we need to convert the boolean matrix to indices

        # We want a vector [N_anchors] with assigned GT index or -1 (background)
        # Initialize with background (-2 for now to distinguish)
        matched_gt_inds = torch.full(
            (num_anchors,), -2, dtype=torch.long, device=anchors.device
        )

        # Find anchors that have at least one positive match
        anchors_with_match = is_pos.any(dim=1)

        if anchors_with_match.any():
            # For these anchors, find the GT with max IoU
            valid_ious = ious[anchors_with_match]
            valid_is_pos = is_pos[anchors_with_match]

            # Mask out non-positive matches in the IoU matrix for argmax
            valid_ious = valid_ious.masked_fill(~valid_is_pos, -1.0)

            max_iou_per_anchor, max_gt_idx = valid_ious.max(dim=1)

            # Assign
            matched_gt_inds[anchors_with_match] = max_gt_idx

        # Background is where matched_gt_inds is still -2.
        # (In ATSS, usually everything not positive is negative/background)
        # So we can just treat -2 as 0 (background class) or handle ignore regions.
        # Here we treat everything else as background.

        return matched_gt_inds, ious


class Criterion(nn.Module):
    """
    Multi-task Loss for Object Detection (ATSS) and Study Classification.
    """

    def __init__(self):
        super(Criterion, self).__init__()
        self.cls_loss = QualityFocalLoss(beta=2.0, reduction="sum")
        self.reg_loss = GIoULoss(reduction="sum")
        self.study_loss = nn.CrossEntropyLoss()
        self.matcher = ATSSMatcher(topk=9)

        self.weight_cls = Config.LOSS_WEIGHT_CLS_DET
        self.weight_box = Config.LOSS_WEIGHT_BOX
        self.weight_study = Config.LOSS_WEIGHT_STUDY

    def decode_boxes(self, anchors, box_preds):
        """
        Decode predicted offsets to x1y1x2y2.
        anchors: [N, 4] (cx, cy, stride, stride)
        box_preds: [N, 4] (l, t, r, b)
        """
        cx, cy, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        l, t, r, b = box_preds[:, 0], box_preds[:, 1], box_preds[:, 2], box_preds[:, 3]

        x1 = cx - l * stride
        y1 = cy - t * stride
        x2 = cx + r * stride
        y2 = cy + b * stride

        return torch.stack([x1, y1, x2, y2], dim=1)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, preds, targets):
        """
        Calculates loss. Forces FP32 to prevent overflow in IoU calculations.
        """
        # Ensure all inputs are float32
        preds = {
            k: v.float() if isinstance(v, torch.Tensor) else v for k, v in preds.items()
        }

        study_logits = preds["study_logits"]
        study_targets = torch.stack([t["study_label"] for t in targets]).to(
            study_logits.device
        )
        loss_study = self.study_loss(study_logits, study_targets)

        cls_logits = preds["cls_logits"]
        bbox_preds = preds["bbox_preds"]
        anchors = preds["anchors"]
        num_anchors_list = preds["num_anchors_per_level"]

        batch_size = cls_logits.size(0)

        loss_cls = 0.0
        loss_box = 0.0
        num_positives = 0

        for i in range(batch_size):
            gt_boxes = targets[i]["boxes"].to(anchors.device).float()
            gt_labels = targets[i]["labels"].to(
                anchors.device
            )  # Usually all 1s for opacity

            # Label Assignment
            matched_gt_inds, ious = self.matcher(anchors, gt_boxes, num_anchors_list)

            # Identify Positives and Negatives
            pos_mask = matched_gt_inds >= 0

            # Prepare Classification Targets (Quality Focal Loss)
            # Target is 0 for neg, IoU for pos
            cls_target = torch.zeros_like(cls_logits[i], dtype=torch.float32)

            if pos_mask.sum() > 0:
                # Assign IoU score as target for positive class
                pos_anchor_inds = torch.where(pos_mask)[0]
                matched_gt_inds_pos = matched_gt_inds[pos_mask]
                pos_ious = ious[pos_anchor_inds, matched_gt_inds_pos]

                # Assuming single class detection (opacity)
                # If multi-class, we would scatter IoUs to specific class indices
                cls_target[pos_mask, 0] = pos_ious

                # Regression Loss (only for positives)
                pred_boxes_pos = bbox_preds[i][pos_mask]
                anchors_pos = anchors[pos_mask]
                gt_boxes_pos = gt_boxes[matched_gt_inds[pos_mask]]

                decoded_pred_boxes = self.decode_boxes(anchors_pos, pred_boxes_pos)

                loss_box += self.reg_loss(decoded_pred_boxes, gt_boxes_pos)
                num_positives += pos_mask.sum().item()

            # Classification Loss (Positives + Negatives)
            loss_cls += self.cls_loss(cls_logits[i], cls_target)

        # Normalize losses
        if num_positives > 0:
            loss_cls /= num_positives
            loss_box /= num_positives
        else:
            # If no positives in entire batch (rare but possible), normalize by batch size or 1
            loss_cls /= batch_size
            # loss_box is 0
            loss_box = torch.tensor(0.0, device=cls_logits.device)

        total_loss = (
            (self.weight_cls * loss_cls)
            + (self.weight_box * loss_box)
            + (self.weight_study * loss_study)
        )

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_study": loss_study,
        }
