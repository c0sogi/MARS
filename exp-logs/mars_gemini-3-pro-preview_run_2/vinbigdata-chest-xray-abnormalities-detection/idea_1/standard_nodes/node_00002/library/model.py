import torch
import torchvision
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetHead

from library.config import Config
from library.utils import get_device


def get_model(pretrained=True):
    """
    Constructs the RetinaNet model with a ResNet-50 FPN backbone.
    The model is adapted to the specific number of classes defined in Config.

    Args:
        pretrained (bool): If True, initializes the model with weights pre-trained on COCO.
                           Defaults to True.

    Returns:
        model (torch.nn.Module): The RetinaNet model moved to the configured device.
    """
    # Select weights based on the pretrained flag
    weights = None
    if pretrained:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT

    # Load the base model with the ResNet-50 FPN backbone
    # This loads the architecture and optionally the pre-trained weights
    model = retinanet_resnet50_fpn_v2(weights=weights)

    # The pre-trained model has a head configured for COCO (91 classes).
    # We need to replace the head to match our dataset (Config.NUM_CLASSES = 15).

    # 1. Get the input channels for the head from the backbone.
    #    For ResNet-50 FPN, out_channels is typically 256.
    in_channels = model.backbone.out_channels

    # 2. Get the number of anchors per spatial location.
    #    We retrieve this from the existing head to ensure it matches the
    #    configuration of the model's anchor generator.
    num_anchors = model.head.classification_head.num_anchors

    # 3. Create a new RetinaNetHead.
    #    This head includes both the classification and regression subnets.
    #    num_classes includes the background class (index 0).
    new_head = RetinaNetHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=Config.NUM_CLASSES
    )

    # Replace the existing head with the new custom head
    model.head = new_head

    # Move the model to the appropriate device (GPU if available)
    device = get_device()
    model.to(device)

    return model
