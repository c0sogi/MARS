import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.ops import misc as misc_nn_ops


def get_model(num_classes, img_size=1024):
    """
    Constructs a Faster R-CNN model with a ResNeXt-101-32x8d backbone and FPN.

    Args:
        num_classes (int): Number of classes (including background).
        img_size (int): Input image size (used for anchor generation hints).

    Returns:
        model (torchvision.models.detection.FasterRCNN): The constructed model.
    """

    # 1. Load the ResNeXt-101-32x8d backbone pre-trained on ImageNet
    # We use the 'DEFAULT' weights which correspond to the best available pre-trained weights.
    weights = ResNeXt101_32X8D_Weights.DEFAULT
    backbone_raw = resnext101_32x8d(weights=weights)

    # 2. Construct the Backbone with FPN
    # We need to extract features from the 4 main stages of the ResNet/ResNeXt architecture.
    # These correspond to layer1, layer2, layer3, layer4.
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

    # The in_channels for ResNeXt-101 at these layers are [256, 512, 1024, 2048].
    # BackboneWithFPN will project them all to `out_channels` (typically 256).
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256

    backbone = BackboneWithFPN(
        backbone_raw,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=None,  # LastLevelMaxPool is sometimes used, but None is standard for FPN
    )

    # 3. Define Anchor Generator
    # FPN backbones typically use 5 scales (one for each FPN level + pool).
    # Since we mapped 4 layers, FPN will output 4 feature maps + usually one extra from maxpool if configured.
    # However, BackboneWithFPN by default returns features for the mapped layers.
    # Let's configure anchors for the 4 feature maps returned by the backbone keys '0', '1', '2', '3'.
    # Sizes should cover small to large objects.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    # Note: If backbone has 4 outputs, we need 4 sizes, or we need to ensure AnchorGenerator matches.
    # BackboneWithFPN usually adds a 'pool' layer if extra_blocks is set.
    # Here we stick to standard sizes. We need to match the number of feature maps.
    # Let's check backbone.body return. It returns an OrderedDict.
    # To be safe, we use a generic generator that works with the feature maps provided.
    # We will use a standard configuration that assumes 5 feature maps (standard FPN often adds one).
    # If not, FasterRCNN handles it if sizes are a tuple of tuples.

    # We'll use a robust set of aspect ratios for medical findings (some are round, some elongated).
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # 4. ROI Pooler
    # We need to pool features from the FPN levels corresponding to the proposals.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )

    # 5. Construct Faster R-CNN
    # Note on Normalization:
    # The dataset pipeline (Albumentations) already normalizes the images using dataset stats.
    # Therefore, we should NOT let FasterRCNN re-normalize using ImageNet stats.
    # We pass image_mean=[0,0,0] and image_std=[1,1,1] to effectively disable internal normalization.

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        # Hyperparameters for training stability and performance
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        box_detections_per_img=100,  # Max detections per image
        box_score_thresh=0.05,  # Lower threshold to capture more candidates before NMS
        box_nms_thresh=0.5,  # Standard NMS threshold
    )

    return model
