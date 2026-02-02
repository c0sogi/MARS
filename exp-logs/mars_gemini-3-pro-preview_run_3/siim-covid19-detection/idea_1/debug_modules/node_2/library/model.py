import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from collections import OrderedDict
from library.config import Config


class MultiTaskFasterRCNN(FasterRCNN):
    def __init__(self):
        """
        Multi-task Faster R-CNN with ResNet18-FPN backbone.
        Performs Object Detection (Opacity) and Study Classification (4 classes).
        """
        # 1. Create Backbone
        # ResNet18 with FPN. Output channels = 256.
        backbone = resnet_fpn_backbone("resnet18", pretrained=True)

        # 2. Initialize FasterRCNN
        # We disable internal normalization (mean=0, std=1) because the Dataset already normalizes.
        # We set min/max size to match the input size to minimize internal resizing artifacts.
        super().__init__(
            backbone,
            num_classes=Config.NUM_DETECTION_CLASSES,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            min_size=Config.IMG_SIZE,
            max_size=Config.IMG_SIZE,
        )

        # 3. Study Classification Head
        # Takes the highest level FPN feature map (256 channels) -> 4 classes
        self.study_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, Config.NUM_STUDY_CLASSES),
        )

    def forward(self, images, targets=None):
        """
        Forward pass for Multi-Task Model.

        Args:
            images (list[Tensor]): List of images.
            targets (list[Dict]): List of targets.

        Returns:
            Training: Dict of losses (RPN, ROI, Study).
            Inference: Tuple (detections, study_probs).
        """
        # 1. Pre-Transform Logic & Target Extraction
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")

            # Extract study labels before transform (which might drop custom keys)
            # targets is a list of dicts
            study_labels = torch.stack([t["study_label"] for t in targets])

            # Standard sanity checks from GeneralizedRCNN
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            f"Expected target boxes to be [N, 4], got {boxes.shape}"
                        )
                else:
                    raise ValueError(
                        f"Expected target boxes to be Tensor, got {type(boxes)}"
                    )

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # 2. Transform (Resize/Pad/Normalize)
        # Note: Normalization is effectively disabled via __init__ params,
        # but Padding/Batching still happens here.
        images, targets = self.transform(images, targets)

        # 3. Backbone Feature Extraction
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 4. RPN (Region Proposal Network)
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 5. ROI Heads (Box Regression & Classification)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # 6. Post-process Detections (Map back to original image size)
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        # 7. Study Classification Branch
        # Use the highest level feature map (lowest resolution) from FPN.
        # FPN returns keys '0', '1', '2', '3'. '3' is the smallest spatial map.
        study_features = features["3"]
        study_logits = self.study_head(study_features)

        # 8. Return Logic
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            # Compute Study Loss
            # Ensure labels are on the correct device
            study_labels = study_labels.to(study_logits.device)
            loss_study = F.cross_entropy(study_logits, study_labels)

            losses["loss_study"] = loss_study * Config.STUDY_LOSS_WEIGHT

            return losses
        else:
            # Inference
            study_probs = F.softmax(study_logits, dim=1)
            return detections, study_probs
