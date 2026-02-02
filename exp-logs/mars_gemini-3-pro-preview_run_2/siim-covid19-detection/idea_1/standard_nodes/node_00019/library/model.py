import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from library.config import Config


class MultiTaskFasterRCNN(FasterRCNN):
    """
    Faster R-CNN with an auxiliary Global Classification Head.
    Cite solution_lesson_node_00008, solution_lesson_node_00013
    """

    def __init__(self, backbone, num_classes=None, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)
        # Global head: Adaptive Pooling -> Flatten -> Linear
        # ResNet50-FPN top feature map has 256 channels
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_classifier = nn.Linear(256, 4)  # 4 study-level classes

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4]",
                    )

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                "expecting the last two dimensions of the Tensor to be H and W",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        # Global Classification Logic
        # FPN returns OrderedDict with keys '0', '1', '2', '3'.
        # '3' corresponds to the highest level (lowest resolution) features.
        global_feat = features["3"]
        global_feat = self.global_avg_pool(global_feat)
        global_feat = torch.flatten(global_feat, 1)
        global_logits = self.global_classifier(global_feat)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            # Compute auxiliary global classification loss
            study_labels = torch.stack([t["study_label"] for t in targets])
            global_loss = F.cross_entropy(global_logits, study_labels)

            # Weighted combination
            losses["loss_global_classifier"] = global_loss * Config.LAMBDA_GLOBAL_CLS
            return losses

        return detections


def get_model(num_classes):
    """
    Initializes Multi-Task Faster R-CNN.
    """
    # Use ResNet50-FPN backbone
    backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT")

    # Initialize Multi-Task model
    # Cite solution_lesson_node_00005: Increase RPN capacity
    model = MultiTaskFasterRCNN(
        backbone,
        num_classes,
        rpn_post_nms_top_n_train=3000,
        box_detections_per_img=200,
    )

    return model
