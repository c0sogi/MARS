import torch
import torch.nn as nn
from library.config import Config
from library.dyhead_modules import SwinBackbone, DyHead
from library.detection_head import ATSSHead, QueryClassifier


class SwinDyHeadNet(nn.Module):
    """
    Unified Model Architecture for COVID-19 Radiography Classification and Detection.

    Components:
    1. Backbone: Swin Transformer (Tiny)
    2. Neck: Dynamic Head (DyHead) with Scale, Spatial, and Task-aware attention.
    3. Detection Head: ATSS (Adaptive Training Sample Selection) for opacity detection.
    4. Study Head: Query-Based Classifier for study-level diagnosis.
    """

    def __init__(self):
        super(SwinDyHeadNet, self).__init__()

        # 1. Backbone
        # Extracts hierarchical features [P3, P4, P5]
        self.backbone = SwinBackbone()

        # 2. Neck (Dynamic Head)
        # Refines features using attention mechanisms
        self.dyhead = DyHead(
            in_channels=self.backbone.out_channels,
            out_channels=Config.DYHEAD_CHANNELS,
            num_blocks=Config.DYHEAD_NUM_BLOCKS,
        )

        # 3. Detection Head (ATSS)
        # Predicts bounding boxes and objectness scores
        self.det_head = ATSSHead(
            in_channels=Config.DYHEAD_CHANNELS,
            num_classes=Config.NUM_CLASSES_DET,
            num_anchors=1,  # Standard for ATSS
        )

        # 4. Study Head (Query Classifier)
        # Predicts the 4 study-level categories
        self.study_head = QueryClassifier(
            in_channels=Config.DYHEAD_CHANNELS, num_classes=Config.NUM_CLASSES_STUDY
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input images of shape [B, 3, H, W]

        Returns:
            dict: A dictionary containing outputs required by the loss function:
                - 'cls_logits': Detection classification logits
                - 'bbox_preds': Bounding box regression offsets
                - 'anchors': Generated anchors
                - 'num_anchors_per_level': Number of anchors per FPN level
                - 'study_logits': Study-level classification logits
        """
        # Backbone: Extract features
        # Returns list of tensors [P3, P4, P5]
        features = self.backbone(x)

        # Neck: Refine features
        # Returns list of tensors [P3, P4, P5] with DYHEAD_CHANNELS
        features = self.dyhead(features)

        # Detection Head
        cls_logits, bbox_preds, anchors, num_anchors_list = self.det_head(features)

        # Study Classification Head
        study_logits = self.study_head(features)

        return {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
            "anchors": anchors,
            "num_anchors_per_level": num_anchors_list,
            "study_logits": study_logits,
        }
