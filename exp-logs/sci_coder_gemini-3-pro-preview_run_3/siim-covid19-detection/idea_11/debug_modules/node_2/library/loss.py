import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou

from library.config import Config
from library.utils import box_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    """
    Modules to compute the matching cost and solve the corresponding LSAP.
    """

    def __init__(
        self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0
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
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets]) # Not used directly for binary class cost approx
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        # For binary (1 class), out_prob[:, 0] is the prob of being opacity.
        # We approximate the focal loss cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        # Cost for class 0 (opacity)
        # shape: (BS*NQ, 1)
        cost_class = pos_cost_class[:, 0:1] - neg_cost_class[:, 0:1]

        # Repeat for the number of targets (since all targets are class 0)
        # Note: In standard multiclass, we would index by tgt_ids.
        # Here we broadcast because all GTs are the same class.
        # We need shape (BS*NQ, Total_GT).
        # cost_class is (BS*NQ, 1). We repeat it Total_GT times.
        cost_class = cost_class.repeat(1, len(tgt_bbox))

        # Compute the L1 cost between boxes
        # out_bbox: (BS*NQ, 4), tgt_bbox: (Total_GT, 4)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the GIoU cost between boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1)

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


class DINOLoss(nn.Module):
    """
    Computes the losses for Multi-Task DINO:
    1. Focal Loss for labels
    2. L1 + GIoU Loss for boxes
    3. Cross Entropy for Study Label
    """

    def __init__(self, matcher, weight_dict, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_classes = Config.NUM_CLASSES

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

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)"""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)

        # Target construction for Sigmoid Focal Loss
        # Target shape: (B, Q, NumClasses) -> (B, Q, 1)
        target_classes_onehot = torch.zeros_like(src_logits)

        # Set matched indices to 1 (opacity)
        target_classes_onehot[idx] = 1.0

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        loss_ce = loss_ce.mean(1).sum() / num_boxes
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss"""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = loss_giou.sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_study(self, outputs, targets, indices=None, num_boxes=None):
        """Compute the study level classification loss"""
        if "study_logits" not in outputs:
            return {}

        src_logits = outputs["study_logits"]
        target_labels = torch.stack([t["study_label"] for t in targets])

        # Standard Cross Entropy
        loss_study = F.cross_entropy(src_logits, target_labels)
        return {"loss_study": loss_study}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "study": self.loss_study,
        }
        assert loss in loss_map, f"do not know {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def prepare_for_dn(self, dn_meta, targets, num_queries):
        """
        Prepare indices for DN components.
        We don't need matching for DN, we construct the indices based on the structure.
        """
        if dn_meta is None:
            return None

        pad_size = dn_meta["pad_size"]
        num_dn_group = dn_meta["num_dn_group"]

        dn_indices = []

        for b, t in enumerate(targets):
            num_gt = len(t["labels"])
            if num_gt > 0:
                # Src indices (predictions)
                src_idx = []
                # Tgt indices (GT)
                tgt_idx = []

                for g in range(num_dn_group):
                    start = g * pad_size
                    # Valid indices in this group (0 to num_gt-1)
                    s = torch.arange(start, start + num_gt, dtype=torch.int64)
                    src_idx.append(s)

                    t_idx = torch.arange(0, num_gt, dtype=torch.int64)
                    tgt_idx.append(t_idx)

                src_idx = torch.cat(src_idx)
                tgt_idx = torch.cat(tgt_idx)

                dn_indices.append((src_idx, tgt_idx))
            else:
                dn_indices.append(
                    (
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64),
                    )
                )

        return dn_indices

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}

        # 1. Main Losses (Detection)
        losses.update(self.get_loss("labels", outputs, targets, indices, num_boxes))
        losses.update(self.get_loss("boxes", outputs, targets, indices, num_boxes))

        # 2. Study Loss
        losses.update(self.get_loss("study", outputs, targets, indices, num_boxes))

        # 3. Auxiliary Losses
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss_name in ["labels", "boxes"]:
                    l_dict = self.get_loss(
                        loss_name, aux_outputs, targets, indices_aux, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 4. DN Losses
        if "dn_logits" in outputs and "dn_meta" in outputs:
            dn_meta = outputs["dn_meta"]
            dn_indices = self.prepare_for_dn(
                dn_meta, targets, outputs["dn_logits"].shape[1]
            )

            # Create a mini-output dict for DN
            dn_outputs = {
                "pred_logits": outputs["dn_logits"],
                "pred_boxes": outputs["dn_boxes"],
            }

            # Compute losses with fixed indices
            # Normalize by num_dn_group to keep scale consistent
            l_dict_labels = self.get_loss(
                "labels",
                dn_outputs,
                targets,
                dn_indices,
                num_boxes * dn_meta["num_dn_group"],
            )
            l_dict_boxes = self.get_loss(
                "boxes",
                dn_outputs,
                targets,
                dn_indices,
                num_boxes * dn_meta["num_dn_group"],
            )

            losses.update({k + "_dn": v for k, v in l_dict_labels.items()})
            losses.update({k + "_dn": v for k, v in l_dict_boxes.items()})

        # 5. Encoder Output Losses (for Two-Stage/Selection)
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            # We treat encoder outputs as predictions and match them
            indices_enc = self.matcher(enc_outputs, targets)
            for loss_name in ["labels", "boxes"]:
                l_dict = self.get_loss(
                    loss_name, enc_outputs, targets, indices_enc, num_boxes
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
