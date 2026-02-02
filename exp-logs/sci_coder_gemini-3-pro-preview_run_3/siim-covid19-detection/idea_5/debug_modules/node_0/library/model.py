import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnext101_32x4d, ResNeXt101_32X4D_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from library.config import Config


class GlobalContextBlock(nn.Module):
    """
    Global Context Block (GCBlock) for capturing long-range dependencies.
    Based on 'GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond'.
    """

    def __init__(self, in_channels, ratio=16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        # Context Modeling: 1x1 Conv + Softmax
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # Transform: 1x1 Conv -> LayerNorm -> ReLU -> 1x1 Conv
        inner_channels = int(in_channels / ratio)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1),
            nn.LayerNorm([inner_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, in_channels, kernel_size=1),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Context Modeling
        input_x = x
        input_x = input_x.view(b, c, h * w).unsqueeze(1)  # [B, 1, C, H*W]

        mask = self.conv_mask(x).view(b, 1, h * w)  # [B, 1, H*W]
        mask = self.softmax(mask).unsqueeze(-1)  # [B, 1, H*W, 1]

        # Global context: [B, 1, C, 1]
        context = torch.matmul(input_x, mask).view(b, c, 1, 1)

        # Transform
        context_transform = self.channel_add_conv(context)

        # Fusion
        return x + context_transform


class StudyClassifier(nn.Module):
    """
    Auxiliary head for study-level classification (Negative, Typical, Indeterminate, Atypical).
    Attaches to the deepest FPN layer.
    """

    def __init__(self, in_channels, num_classes):
        super(StudyClassifier, self).__init__()
        self.gc_block = GlobalContextBlock(in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x is the feature map from the deepest FPN layer (e.g., P5)
        x = self.gc_block(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C], targets: [B]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CovidDetector(FasterRCNN):
    """
    Multi-Task Detector for COVID-19 Radiographs.

    Architecture:
    - Backbone: ResNeXt-101 (32x4d) with FPN.
    - Detection Head: Faster R-CNN (RPN + RoI Heads).
    - Study Head: Global Context Block + Classifier.
    """

    def __init__(self, num_object_classes, num_study_classes):
        # 1. Build Backbone: ResNeXt-101-32x4d + FPN
        backbone_model = resnext101_32x4d(weights=ResNeXt101_32X4D_Weights.DEFAULT)

        # Extract layers for FPN (layer1..4)
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

        # Channel counts for ResNeXt-101
        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256

        backbone = BackboneWithFPN(
            backbone_model,
            return_layers,
            in_channels_list,
            out_channels,
            extra_blocks=None,
        )

        # 2. Initialize FasterRCNN
        super(CovidDetector, self).__init__(backbone, num_classes=num_object_classes)

        # 3. Study Classification Head
        # Takes the deepest FPN feature ('3') which has 'out_channels' (256)
        self.study_classifier = StudyClassifier(out_channels, num_study_classes)

        # 4. Loss Function for Study Head
        self.study_criterion = FocalLoss(gamma=2.0)

    def forward(self, images, targets=None):
        """
        Forward pass handling both detection and study classification.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Standard FasterRCNN preprocessing (normalize, resize if needed)
        # images is a list of tensors, targets is a list of dicts
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # 1. Backbone Forward
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 2. RPN Forward
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 3. RoI Heads Forward (Detection)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # 4. Study Head Forward
        # Use the deepest feature map (key '3' corresponds to layer4 output in FPN)
        # If FPN keys are different, we take the last one.
        deepest_layer_key = list(features.keys())[-1]
        study_features = features[deepest_layer_key]
        study_logits = self.study_classifier(study_features)

        # 5. Output Construction
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            # Calculate Study Loss
            if targets is not None:
                study_targets = torch.stack([t["study_ids"] for t in targets])
                study_loss = self.study_criterion(study_logits, study_targets)
                losses["loss_study"] = study_loss * Config.LOSS_WEIGHT_STUDY

            return losses
        else:
            # Inference Mode
            # Detections are already computed by roi_heads
            # Add study predictions to the result
            study_probs = F.softmax(study_logits, dim=1)

            # Attach study predictions to the detection results
            # We can't easily modify the list of dicts returned by roi_heads in place without copying
            # But we can return a tuple or modify the dicts.

            results = []
            for i, det in enumerate(detections):
                det_res = det.copy()
                det_res["study_probs"] = study_probs[i]
                det_res["study_label"] = torch.argmax(study_probs[i])
                results.append(det_res)

            return results


def get_model():
    """
    Factory function to create the model.
    """
    model = CovidDetector(
        num_object_classes=Config.NUM_OBJECT_CLASSES,
        num_study_classes=Config.NUM_STUDY_CLASSES,
    )
    return model
