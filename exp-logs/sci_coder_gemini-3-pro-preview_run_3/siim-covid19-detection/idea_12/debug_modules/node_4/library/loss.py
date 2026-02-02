import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss, box_convert

from library.config import Config


class HungarianMatcher(nn.Module):
    """
    Modules to compute the matching cost and solve the corresponding LSAP.
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
        Performs the matching.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
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
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        # For binary classification (opacity vs background), num_classes is 1.
        # Focal loss cost: alpha * (1 - p)^gamma * -log(p)
        # We approximate the cost using the probability of the target class.
        # Since we only have one class 'opacity', we look at the prob of that class.
        # Note: This is a simplified cost calculation suitable for matching.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        # If tgt_ids is all 0 (class 0), we want the cost associated with predicting class 0.
        # However, typically in DETR/DINO, 'labels' contains the class index.
        # Here we have 1 class (index 0).
        # We construct the cost matrix based on the probability of the specific target class.
        # Since all targets are class 0, we just take the column 0 of the cost.
        cost_class = pos_cost_class[:, 0:1] - neg_cost_class[:, 0:1]

        # If we had multiple classes, we would index using tgt_ids.
        # But here we repeat the cost vector for each target since they are all the same class.
        # To handle variable number of targets per image, we need to be careful.
        # Actually, the standard way is to gather the costs for the specific target labels.
        # Since all our targets are class 0, we just pick column 0.
        # But we need to shape it to [batch*queries, total_num_targets]
        # This logic simplifies if we just compute cost for class 0 for all targets.

        # Let's stick to the standard implementation logic:
        # cost_class = -out_prob[:, tgt_ids] # Simple probability maximization
        # Or focal loss cost.

        # Re-implementing standard DETR/DINO focal loss cost for matching:
        # out_prob: [N_queries, 1]
        # tgt_ids: [N_targets] (all 0s)
        # We want cost matrix [N_queries, N_targets]
        # For each target, the cost is the same column from out_prob because they are all class 0.
        cost_class = cost_class.repeat(1, len(tgt_ids))

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        # box_convert: cxcywh -> xyxy
        out_bbox_xyxy = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        tgt_bbox_xyxy = box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

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


class MultiTaskCriterion(nn.Module):
    """
    This class computes the loss for MultiTaskDINO.
    The process happens in two steps:
        1. We compute hungarian assignment between ground truth boxes and the outputs of the model
        2. We supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self):
        super().__init__()
        self.matcher = HungarianMatcher(
            cost_class=Config.COST_CLASS,
            cost_bbox=Config.COST_BBOX,
            cost_giou=Config.COST_GIOU,
        )
        self.weight_dict = {
            "loss_ce": Config.COST_CLASS,
            "loss_bbox": Config.COST_BBOX,
            "loss_giou": Config.COST_GIOU,
            "loss_study": Config.LOSS_STUDY_WEIGHT,
        }
        self.detection_weight = Config.LOSS_DETECTION_WEIGHT

        # Focal loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Focal Loss)
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)

        # Prepare target classes
        # Shape: [batch_size, num_queries, num_classes]
        # We initialize with 0 (background/negative)
        # In Sigmoid Focal Loss, we provide binary targets for each class.
        # Here num_classes = 1.
        target_classes_onehot = torch.zeros_like(src_logits)

        # Set matched indices to 1 (positive class)
        # indices contains (batch_idx, src_idx)
        # We set target_classes_onehot[batch_idx, src_idx, 0] = 1
        target_classes_onehot[idx] = 1.0

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        loss_ce = loss_ce.sum() / num_boxes
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        # Select the predicted boxes that matched a ground truth
        src_boxes = outputs["pred_boxes"][idx]

        # Select the corresponding target boxes
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        # Compute GIoU loss
        # Convert cxcywh to xyxy for GIoU
        src_boxes_xyxy = box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        target_boxes_xyxy = box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")

        loss_giou = 1 - torch.diag(
            generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
        )
        loss_giou = loss_giou.sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_study(self, outputs, targets):
        """
        Compute the study-level classification loss (Cross Entropy).
        """
        assert "study_logits" in outputs
        src_logits = outputs["study_logits"]

        # Concatenate study labels from targets
        target_labels = torch.cat([t["study_labels"] for t in targets])

        loss_study = F.cross_entropy(src_logits, target_labels)
        return {"loss_study": loss_study}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # In distributed training, we would sync here. For single GPU, just clamp.
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}

        # 1. Detection Losses
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # 2. Study Loss
        losses.update(self.loss_study(outputs, targets))

        # 3. Weighted Sum
        total_loss = (
            self.detection_weight
            * (
                self.weight_dict["loss_ce"] * losses["loss_ce"]
                + self.weight_dict["loss_bbox"] * losses["loss_bbox"]
                + self.weight_dict["loss_giou"] * losses["loss_giou"]
            )
            + self.weight_dict["loss_study"] * losses["loss_study"]
        )

        losses["total_loss"] = total_loss

        return losses
