import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from library.config import Config
from library.utils import box_cxcywh_to_xyxy, box_iou


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format.
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou = box_iou(boxes1, boxes2)

    # Re-calculate areas and intersection to get union area for enclosing box calc
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    # Smallest Enclosing Box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25 (for positive).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
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

        # We get the predicted logits for the object class(es)
        # We slice to ignore the background/extra channel if present, as we use Sigmoid Focal Loss
        # shape: [batch_size * num_queries, num_object_classes]
        out_prob = (
            outputs["pred_logits"][..., : Config.NUM_OBJECT_CLASSES]
            .flatten(0, 1)
            .sigmoid()
        )

        # shape: [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        # For focal loss, the cost is roughly: -alpha * (1-p)^gamma * log(p)
        # We use the probability of the specific target class.
        # Since we have effectively 1 class (opacity, index 0), we extract prob for class 0.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        # Construct cost matrix for classification
        # tgt_ids contains class indices (all 0 for opacity).
        # We gather the cost for the specific class.
        # Since we only have 1 class, out_prob is (N, 1).
        # cost_class shape: (Batch*Queries, Total_Targets)
        # We want cost for assigning query i to target j (which is class c_j)
        # cost = pos_cost_class[:, c_j] - neg_cost_class[:, c_j]
        # This formulation balances the fact that if we don't pick it, we pay neg_cost.
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # box_cxcywh_to_xyxy is needed because giou expects xyxy
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


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1. we compute hungarian assignment between ground truth boxes and the outputs of the model
        2. we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self):
        """Create the criterion."""
        super().__init__()
        self.num_classes = Config.NUM_OBJECT_CLASSES
        self.matcher = HungarianMatcher(
            cost_class=Config.COST_CLASS,
            cost_bbox=Config.COST_BBOX,
            cost_giou=Config.COST_GIOU,
        )

        # Weights
        self.weight_dict = {
            "loss_ce": Config.LOSS_CLASS,
            "loss_bbox": Config.LOSS_BBOX,
            "loss_giou": Config.LOSS_GIOU,
            "loss_study": Config.LOSS_CE_STUDY,
        }

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain "labels" containing a tensor of dim [num_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)

        # Prepare targets
        # Initialize with background (0 for one-hot if we considered background as 0,
        # but for sigmoid focal loss, target is 0 or 1 per class).
        # We create a target tensor of shape (B, Q, Num_Classes) filled with 0.
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], self.num_classes],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )

        # Get the target classes for the matched indices
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        # Assign 1.0 to the matched locations
        # target_classes_o contains class indices (all 0 for opacity).
        target_classes_onehot[idx[0], idx[1], target_classes_o] = 1.0

        # Compute Sigmoid Focal Loss
        # We only use the first num_classes channels of the prediction
        src_logits_focal = src_logits[..., : self.num_classes]

        loss_ce = sigmoid_focal_loss(
            src_logits_focal, target_classes_onehot, num_boxes, alpha=0.25, gamma=2
        )

        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain "boxes" containing a tensor of dim [num_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        if len(target_boxes) == 0:
            return {
                "loss_bbox": torch.tensor(0.0).to(src_boxes.device),
                "loss_giou": torch.tensor(0.0).to(src_boxes.device),
            }

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_study(self, outputs, targets):
        """Compute the study-level classification loss"""
        assert "pred_study_logits" in outputs
        src_study = outputs["pred_study_logits"]

        # Stack study labels from targets
        target_study = torch.stack([t["study_label"] for t in targets])

        loss_study = F.cross_entropy(src_study, target_study)
        return {"loss_study": loss_study}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # Clamp to 1 to avoid division by zero if no boxes in batch
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}

        # 1. Object Detection Losses
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # 2. Study Classification Loss
        losses.update(self.loss_study(outputs, targets))

        # 3. Weighted Sum
        final_loss = (
            losses["loss_ce"] * self.weight_dict["loss_ce"]
            + losses["loss_bbox"] * self.weight_dict["loss_bbox"]
            + losses["loss_giou"] * self.weight_dict["loss_giou"]
            + losses["loss_study"] * self.weight_dict["loss_study"]
        )

        losses["loss"] = final_loss

        return losses
