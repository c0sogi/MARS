import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights


def get_kuzushiji_model(num_classes, min_size=1024, max_size=2048):
    """
    Initializes a Faster R-CNN model with a ResNet-50-FPN backbone.
    Configured specifically for the Kuzushiji recognition task.

    Args:
        num_classes (int): The total number of classes (including background).
        min_size (int): Minimum size of the image to be rescaled to.
        max_size (int): Maximum size of the image to be rescaled to.

    Returns:
        model (torchvision.models.detection.FasterRCNN): The configured model.
    """
    # Ensure reproducibility for model initialization
    torch.manual_seed(42)

    # 1. Construct the Backbone
    # Cite Lesson 00026: Prefer lighter backbones (ResNet50) for geometric primitives.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT
    )

    # 2. Initialize the Faster R-CNN
    # Cite Lesson 00015: Inject preprocessing parameters (min_size, max_size) via constructor.
    # Cite Lesson 00005: Match input resolution to object granularity (1024/2048).
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        # RPN: Keep 2000 proposals after NMS during testing (Cite Lesson 00016)
        rpn_post_nms_top_n_test=2000,
        # ROI Heads: Allow up to 1200 detections per image (Cite Lesson 00008)
        box_detections_per_img=1200,
        # ROI Heads: Set score threshold to 0.35 (Cite Lesson 00013)
        box_score_thresh=0.35,
    )

    return model
