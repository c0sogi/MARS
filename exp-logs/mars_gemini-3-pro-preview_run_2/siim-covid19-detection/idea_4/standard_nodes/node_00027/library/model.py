import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights

from library.config import Config


class SpatialAttentionHead(nn.Module):
    """
    Multi-Scale Spatial Attention Head for Study Classification.
    Extracts features from FPN levels P3, P4, P5, applies spatial attention,
    and performs global classification.
    """

    def __init__(self, in_channels=256, num_classes=4):
        super().__init__()
        # Attention mechanisms for FPN level P5 (key '3') only
        # Cite solution_lesson_node_00026: Decouple high-resolution features from global classification
        self.att_convs = nn.ModuleDict(
            {
                "3": nn.Conv2d(in_channels, 1, kernel_size=1),
            }
        )

        # Classifier: Features from P5 -> num_classes
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, features):
        """
        Args:
            features (OrderedDict): Output from FPN backbone.
        Returns:
            logits (Tensor): (N, num_classes)
        """
        pooled_feats = []
        # Process specific FPN level: P5
        # Keys in resnet_fpn_backbone are '0' (P2), '1' (P3), '2' (P4), '3' (P5)
        target_keys = ["3"]

        for key in target_keys:
            if key in features:
                x = features[key]  # (N, C, H, W)

                # Generate Spatial Attention Map
                att_logit = self.att_convs[key](x)  # (N, 1, H, W)
                att_map = torch.sigmoid(att_logit)

                # Apply Attention
                x_weighted = x * att_map

                # Global Average Pooling
                # Result: (N, C)
                x_pooled = x_weighted.mean(dim=(2, 3))
                pooled_feats.append(x_pooled)
            else:
                # Should not happen with standard ResNet-FPN
                continue

        if not pooled_feats:
            return None

        # Concatenate features (only one level now): (N, C)
        concat_feats = torch.cat(pooled_feats, dim=1)

        # Classification
        logits = self.classifier(concat_feats)
        return logits


class MultiTaskFasterRCNN(FasterRCNN):
    """
    Faster R-CNN with an integrated Spatial Attention Head for multi-task learning.
    """

    def __init__(self, backbone, num_classes, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)

        # Initialize Auxiliary Head
        # FPN output channels are typically 256
        self.study_head = SpatialAttentionHead(
            in_channels=backbone.out_channels, num_classes=Config.NUM_STUDY_CLASSES
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 1. Transform (Resize, Normalize, etc.)
        # Note: We disable internal normalization in get_model, so this primarily handles resizing/batching
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # 2. Backbone Forward
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 3. RPN & ROI Heads (Object Detection)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # 4. Spatial Attention Head (Study Classification)
        study_logits = self.study_head(features)

        # 5. Output Construction
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            # Calculate Study Loss
            # targets is a list of dicts, 'study_label' is preserved by transform
            gt_study_labels = torch.stack([t["study_label"] for t in targets])

            study_loss = F.cross_entropy(study_logits, gt_study_labels)

            # Weighted Auxiliary Loss
            losses["loss_study"] = study_loss * Config.AUX_LOSS_WEIGHT

            return losses
        else:
            # Inference: Return detections and study probabilities
            study_probs = F.softmax(study_logits, dim=1)
            return detections, study_probs


def get_model():
    """
    Constructs the Multi-Task Faster R-CNN model with ResNet-101-FPN backbone.
    Configures RPN and ROI heads according to the strategy.
    """
    # 1. Backbone: ResNet-101-FPN
    # We use pretrained weights for better convergence
    backbone = resnet_fpn_backbone(
        Config.BACKBONE, weights=ResNet101_Weights.IMAGENET1K_V1, trainable_layers=3
    )

    # 2. Instantiate Model
    # We explicitly disable internal normalization (mean=0, std=1) because
    # the dataset loader already applies ImageNet normalization.
    model = MultiTaskFasterRCNN(
        backbone,
        num_classes=Config.NUM_DETECTION_CLASSES,
        # Pre-processing config
        min_size=Config.IMAGE_SIZE,
        max_size=Config.IMAGE_SIZE,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        # RPN Config (High Capacity)
        rpn_pre_nms_top_n_train=Config.RPN_PRE_NMS_TOP_N_TRAIN,
        rpn_post_nms_top_n_train=Config.RPN_POST_NMS_TOP_N_TRAIN,
        rpn_pre_nms_top_n_test=Config.RPN_PRE_NMS_TOP_N_TEST,
        rpn_post_nms_top_n_test=Config.RPN_POST_NMS_TOP_N_TEST,
        # ROI Heads Config
        box_detections_per_img=Config.DETECTIONS_PER_IMG,
        box_score_thresh=Config.ROI_HEADS_SCORE_THRESH,
        box_nms_thresh=Config.ROI_HEADS_NMS_THRESH,
    )

    return model
