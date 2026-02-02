import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, box_iou
from scipy.optimize import linear_sum_assignment
from library.config import Config

# =========================================================================
# Box Utilities
# =========================================================================


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# =========================================================================
# Matcher
# =========================================================================


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes]
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where each label is the class index)
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        # We use the probability of the target class.
        # alpha = 0.25, gamma = 2.0 are standard focal loss params
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        # Pick the cost for the specific target class
        # tgt_ids are indices of the ground truth classes
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


# =========================================================================
# Loss Module
# =========================================================================


class CoDETRLoss(nn.Module):
    def __init__(self, matcher, num_classes, weight_dict, eos_coef=0.1, losses=None):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses if losses is not None else ["labels", "boxes", "study"]

        # Aux ATSS Config
        self.strides = [8, 16, 32]  # Standard for Swin-L (C3, C4, C5)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Prepare for sigmoid focal loss
        # One-hot encoding for targets
        # src_logits: [B, Q, num_classes + 1] (usually) or [B, Q, num_classes]
        # Here Config.NUM_CLASSES = 1.
        # If we use sigmoid focal loss, we usually project to num_classes (1) and use binary targets.

        # Flatten
        src_logits = src_logits.flatten(0, 1)  # [B*Q, C]
        target_classes = target_classes.flatten(0, 1)  # [B*Q]

        # Prepare targets for focal loss (B*Q, NumClasses)
        # Class self.num_classes is background
        target_one_hot = torch.zeros(
            (len(target_classes), self.num_classes), device=src_logits.device
        )

        # Fill ones where target is not background
        fg_mask = target_classes != self.num_classes
        if fg_mask.any():
            # In this dataset, class index 0 is 'opacity'.
            # If target_classes[i] == 0, we set target_one_hot[i, 0] = 1
            # We only have 1 class, so we just take the indices where it is 0
            target_one_hot[fg_mask, target_classes[fg_mask]] = 1.0

        # We take the logits for the classes (excluding explicit background channel if it exists,
        # but usually DETR output dim is num_classes+1 or num_classes.
        # Config says num_classes=1. Model output is num_classes+1.
        # We only care about the first num_classes channels for focal loss against 0/1 targets.
        src_logits_sig = src_logits[:, : self.num_classes]

        loss_ce = sigmoid_focal_loss(
            src_logits_sig, target_one_hot, alpha=0.25, gamma=2.0, reduction="mean"
        )

        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        loss_giou = loss_giou.sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_study(self, outputs, targets, indices, num_boxes):
        """Compute the study-level classification loss"""
        assert "pred_study" in outputs
        src_study = outputs["pred_study"]  # [B, 4]
        target_study = torch.cat(
            [t["study_label"].unsqueeze(0) for t in targets]
        )  # [B]

        loss_study = F.cross_entropy(src_study, target_study)
        return {"loss_study": loss_study}

    def loss_aux_atss(self, outputs, targets, indices, num_boxes):
        """
        Computes loss for the Auxiliary ATSS Head.
        Uses a simplified Center Sampling strategy.
        """
        enc_outputs = outputs.get("enc_outputs")
        if enc_outputs is None or "atss" not in enc_outputs:
            return {}

        atss_out = enc_outputs["atss"]
        logits_list = atss_out["logits"]  # List of [B, A*C, H, W]
        bbox_regs_list = atss_out["bbox_regs"]  # List of [B, A*4, H, W]
        centerness_list = atss_out["centerness"]  # List of [B, A*1, H, W]

        loss_cls = 0.0
        loss_reg = 0.0
        loss_cnt = 0.0
        n_levels = len(logits_list)

        # Flatten batch
        for b_idx, target in enumerate(targets):
            gt_boxes = target["boxes"]  # cx, cy, w, h (normalized)
            gt_labels = target["labels"]  # class indices

            # If no objects, push all to background
            if len(gt_boxes) == 0:
                for l in range(n_levels):
                    # All background
                    # logits: [B, C, H, W] -> select batch -> [C, H, W]
                    feat_logits = logits_list[l][b_idx]
                    # Target is all zeros
                    target_zeros = torch.zeros_like(feat_logits)
                    # Focal loss
                    loss_cls += sigmoid_focal_loss(
                        feat_logits.permute(1, 2, 0).flatten(0, 1),
                        target_zeros.permute(1, 2, 0).flatten(0, 1),
                        alpha=0.25,
                        gamma=2.0,
                        reduction="sum",
                    )
                continue

            # Assign GT to levels based on scale (heuristic)
            # Area relative to image
            areas = gt_boxes[:, 2] * gt_boxes[:, 3]  # w * h

            # Map each GT to the best level
            # Level 0 (stride 8): small
            # Level 1 (stride 16): medium
            # Level 2 (stride 32): large
            # Simple heuristic: sqrt(area) * 1024 (image size)
            # sqrt(area) is normalized size.
            # s8: < 64px, s16: 64-128px, s32: > 128px
            # normalized: < 0.0625, 0.0625-0.125, > 0.125

            sz = torch.sqrt(areas)
            target_levels = (
                torch.floor(torch.log2(sz / 0.03 + 1e-6))
                .long()
                .clamp(min=0, max=n_levels - 1)
            )

            for l in range(n_levels):
                stride = self.strides[l]
                feat_logits = logits_list[l][b_idx]  # [C, H, W]
                feat_reg = bbox_regs_list[l][b_idx]  # [4, H, W]
                feat_cnt = centerness_list[l][b_idx]  # [1, H, W]

                _, H, W = feat_logits.shape

                # Create targets
                target_cls = torch.zeros(
                    (H, W, self.num_classes), device=feat_logits.device
                )
                target_reg_mask = torch.zeros(
                    (H, W), dtype=torch.bool, device=feat_logits.device
                )
                target_reg = torch.zeros((H, W, 4), device=feat_logits.device)
                target_cnt = torch.zeros((H, W, 1), device=feat_logits.device)

                # Find GTs assigned to this level
                gt_indices = (target_levels == l).nonzero(as_tuple=False).squeeze(1)

                if len(gt_indices) > 0:
                    curr_boxes = gt_boxes[gt_indices]
                    curr_labels = gt_labels[gt_indices]

                    # Project to feature map
                    cx = curr_boxes[:, 0] * W
                    cy = curr_boxes[:, 1] * H
                    w = curr_boxes[:, 2] * W
                    h = curr_boxes[:, 3] * H

                    # Determine grid cells
                    # Center sampling: 3x3 region around center or just center
                    # Use just center for simplicity and high precision
                    gx = cx.long().clamp(0, W - 1)
                    gy = cy.long().clamp(0, H - 1)

                    # Assign
                    for k, (ix, iy) in enumerate(zip(gx, gy)):
                        # Classification
                        # curr_labels[k] is 0 for opacity.
                        target_cls[iy, ix, 0] = 1.0

                        # Regression
                        # Model predicts exp(scale * x). We interpret this as distances l, t, r, b
                        # Target l, t, r, b normalized by stride?
                        # Actually, we can just use the decoded box for GIoU loss directly.
                        # But here we need to store the target for loss calculation.
                        # We will compute GIoU on the fly for positive samples.

                        # Store GT box for this pixel
                        # Convert cx, cy, w, h (feature scale) to l, t, r, b
                        # l = x - (cx - w/2) = w/2 ... wait.
                        # l = grid_x - x_min, t = grid_y - y_min, etc.
                        # x_min = cx_box - w/2

                        l_val = (ix + 0.5) - (cx[k] - w[k] / 2)
                        t_val = (iy + 0.5) - (cy[k] - h[k] / 2)
                        r_val = (cx[k] + w[k] / 2) - (ix + 0.5)
                        b_val = (cy[k] + h[k] / 2) - (iy + 0.5)

                        target_reg[iy, ix] = torch.stack([l_val, t_val, r_val, b_val])
                        target_reg_mask[iy, ix] = True

                        # Centerness
                        # sqrt( (min(l,r)/max(l,r)) * (min(t,b)/max(t,b)) )
                        lr = torch.stack([l_val, r_val])
                        tb = torch.stack([t_val, b_val])
                        cnt_val = torch.sqrt(
                            (lr.min() / lr.max().clamp(min=1e-6))
                            * (tb.min() / tb.max().clamp(min=1e-6))
                        )
                        target_cnt[iy, ix] = cnt_val

                # Compute Losses for this level
                # 1. Classification (Focal)
                # feat_logits: [C, H, W] -> [H, W, C]
                pred_cls = feat_logits.permute(1, 2, 0)
                # Only use first channel if C > 1, but here C=1 (opacity)
                # ATSSHead outputs num_classes * num_anchors. num_anchors=1.
                loss_cls += sigmoid_focal_loss(
                    pred_cls.flatten(0, 1),
                    target_cls.flatten(0, 1),
                    alpha=0.25,
                    gamma=2.0,
                    reduction="sum",
                )

                # 2. Regression (GIoU) & Centerness
                if target_reg_mask.any():
                    # Select positives
                    pos_pred_reg = feat_reg.permute(1, 2, 0)[
                        target_reg_mask
                    ]  # [N_pos, 4]
                    pos_target_reg = target_reg[target_reg_mask]  # [N_pos, 4]

                    # Decode to xyxy for GIoU
                    # Pred is l, t, r, b
                    # Coords are relative to grid center
                    # We can treat grid center as (0,0) for GIoU calc or reconstruction
                    # Reconstruct boxes in feature map coords
                    # x1 = -l, y1 = -t, x2 = r, y2 = b (relative to center)

                    pred_l, pred_t, pred_r, pred_b = pos_pred_reg.unbind(-1)
                    pred_boxes = torch.stack([-pred_l, -pred_t, pred_r, pred_b], dim=1)

                    tgt_l, tgt_t, tgt_r, tgt_b = pos_target_reg.unbind(-1)
                    tgt_boxes = torch.stack([-tgt_l, -tgt_t, tgt_r, tgt_b], dim=1)

                    loss_reg += (
                        1 - torch.diag(generalized_box_iou(pred_boxes, tgt_boxes))
                    ).sum()

                    # Centerness (BCE)
                    pos_pred_cnt = feat_cnt.permute(1, 2, 0)[target_reg_mask]
                    pos_target_cnt = target_cnt[target_reg_mask]
                    loss_cnt += F.binary_cross_entropy_with_logits(
                        pos_pred_cnt, pos_target_cnt, reduction="sum"
                    )

        # Normalize by number of boxes (or batch size) to keep scale consistent
        num_pos = sum(len(t["boxes"]) for t in targets)
        num_pos = max(num_pos, 1)

        return {
            "loss_atss_cls": loss_cls / num_pos,
            "loss_atss_reg": loss_reg / num_pos,
            "loss_atss_cnt": loss_cnt / num_pos,
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # (In a distributed setting we would reduce here, but single GPU is fine)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Compute Aux ATSS Loss
        if "enc_outputs" in outputs and outputs["enc_outputs"] is not None:
            aux_losses = self.loss_aux_atss(outputs, targets, indices, num_boxes)
            losses.update(aux_losses)

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        if loss == "labels":
            return self.loss_labels(outputs, targets, indices, num_boxes)
        elif loss == "boxes":
            return self.loss_boxes(outputs, targets, indices, num_boxes)
        elif loss == "study":
            return self.loss_study(outputs, targets, indices, num_boxes)
        return {}


def build_criterion(config=Config):
    matcher = HungarianMatcher(
        cost_class=config.COST_CLASS,
        cost_bbox=config.COST_BBOX,
        cost_giou=config.COST_GIOU,
    )

    weight_dict = {
        "loss_ce": config.LAMBDA_DETR * config.COST_CLASS,
        "loss_bbox": config.LAMBDA_DETR * config.COST_BBOX,
        "loss_giou": config.LAMBDA_DETR * config.COST_GIOU,
        "loss_study": config.LAMBDA_STUDY,
        "loss_atss_cls": config.LAMBDA_AUX_ATSS,
        "loss_atss_reg": config.LAMBDA_AUX_ATSS,
        "loss_atss_cnt": config.LAMBDA_AUX_ATSS,
    }

    criterion = CoDETRLoss(
        matcher=matcher,
        num_classes=config.NUM_CLASSES,
        weight_dict=weight_dict,
        losses=["labels", "boxes", "study"],
    )
    return criterion
