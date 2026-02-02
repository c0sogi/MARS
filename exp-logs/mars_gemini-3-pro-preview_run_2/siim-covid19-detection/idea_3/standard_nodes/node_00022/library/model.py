import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from library.config import Config


class MultiScaleAttentionHead(nn.Module):
    """
    Auxiliary head that applies spatial attention to multiple FPN levels
    to predict study-level labels.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Attention mechanisms for P3, P4, P5 (keys '1', '2', '3')
        self.att_convs = nn.ModuleDict(
            {
                "1": nn.Conv2d(in_channels, 1, 1),
                "2": nn.Conv2d(in_channels, 1, 1),
                "3": nn.Conv2d(in_channels, 1, 1),
            }
        )

        # Classifier: Concatenates pooled features from 3 levels
        self.classifier = nn.Linear(in_channels * 3, num_classes)

    def forward(self, features):
        """
        Args:
            features (OrderedDict): Output from FPN backbone.
        Returns:
            logits (Tensor): (B, num_classes)
        """
        embeddings = []
        # We use levels '1' (stride 8), '2' (stride 16), '3' (stride 32)
        target_keys = ["1", "2", "3"]

        for key in target_keys:
            if key in features:
                x = features[key]  # (B, C, H, W)

                # Spatial Attention: 1x1 Conv -> Sigmoid
                att_map = torch.sigmoid(self.att_convs[key](x))

                # Apply attention
                x_att = x * att_map

                # Global Average Pooling
                x_pool = x_att.mean(dim=(2, 3))  # (B, C)
                embeddings.append(x_pool)
            else:
                # Fallback for safety, though keys should exist in standard FPN
                # Assuming batch size is consistent with other features
                ref = features[list(features.keys())[0]]
                embeddings.append(
                    torch.zeros(ref.shape[0], ref.shape[1], device=ref.device)
                )

        # Concatenate multi-scale features
        cat_emb = torch.cat(embeddings, dim=1)  # (B, C * 3)
        logits = self.classifier(cat_emb)

        return logits


def get_resnext_fpn_backbone():
    """
    Constructs a ResNeXt-101-32x8d backbone with FPN.
    """
    # Load pretrained ResNeXt-101
    backbone = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)

    # Extract layers for FPN
    # layer1: stride 4, layer2: stride 8, layer3: stride 16, layer4: stride 32
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

    # Input channels for each layer in ResNeXt-101
    in_channels_list = [256, 512, 1024, 2048]

    # Wrap with FPN
    # out_channels=256 is standard for FPN
    backbone_fpn = BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels=256
    )

    return backbone_fpn


class CovidMultiTaskModel(FasterRCNN):
    """
    Multi-Task Faster R-CNN with ResNeXt-FPN and Multi-Scale Attention Head.
    Predicts both bounding boxes (opacities) and study-level labels.
    """

    def __init__(self):
        # 1. Backbone
        backbone = get_resnext_fpn_backbone()

        # 2. Initialize Faster R-CNN
        super().__init__(
            backbone,
            num_classes=Config.NUM_BOX_CLASSES,
            # RPN Parameters
            rpn_pre_nms_top_n_train=Config.RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_train=Config.RPN_POST_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=Config.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_test=Config.RPN_POST_NMS_TOP_N_TEST,
            # Box Head Parameters
            box_detections_per_img=Config.BOX_DETECTIONS_PER_IMG,
            box_score_thresh=Config.BOX_SCORE_THRESH,
            box_nms_thresh=Config.BOX_NMS_THRESH,
            # Preprocessing:
            # Dataset already normalizes and resizes.
            # We set mean/std to identity and min/max size to fixed size to avoid double processing.
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            min_size=Config.IMG_SIZE,
            max_size=Config.IMG_SIZE,
        )

        # 3. Auxiliary Attention Head
        # FPN outputs 256 channels
        self.study_head = MultiScaleAttentionHead(
            in_channels=256, num_classes=Config.NUM_STUDY_CLASSES
        )

    def forward(self, images, targets=None):
        """
        Custom forward pass to include auxiliary head logic.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 1. Transform (Standard FasterRCNN logic)
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # 2. Backbone Features
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 3. Auxiliary Head Forward
        study_logits = self.study_head(features)

        # 4. RPN and ROI Heads (Standard FasterRCNN logic)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # 5. Output Handling
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            # Compute Study Loss
            # targets is List[Dict], need to extract labels
            gt_study_labels = torch.stack([t["study_label"] for t in targets])
            study_loss = F.cross_entropy(study_logits, gt_study_labels)

            # Weighted sum
            losses["loss_study"] = study_loss * Config.GLOBAL_HEAD_LOSS_WEIGHT

            return losses
        else:
            # Inference
            study_probs = F.softmax(study_logits, dim=1)

            # Attach study predictions to detections
            for i, det in enumerate(detections):
                det["study_prediction"] = study_probs[i]

            return detections
