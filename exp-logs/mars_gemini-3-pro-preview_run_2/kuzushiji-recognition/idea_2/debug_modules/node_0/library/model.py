import torch
import torchvision
from torchvision.models.detection import CascadeRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights


def get_kuzushiji_model(num_classes):
    """
    Initializes a Cascade R-CNN model with a ResNet-101-FPN backbone.
    Configured specifically for the Kuzushiji recognition task.

    Args:
        num_classes (int): The total number of classes (including background).
                           For this dataset, it should be 3848 + 1 = 3849.

    Returns:
        model (torchvision.models.detection.CascadeRCNN): The configured model.
    """
    # Ensure reproducibility for model initialization
    torch.manual_seed(42)

    # 1. Construct the Backbone
    # We use ResNet-101 with FPN, initialized with default ImageNet weights.
    # trainable_layers=3 is the default, which freezes the first 2 blocks (conv1, layer1).
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101", weights=ResNet101_Weights.DEFAULT
    )

    # 2. Initialize the Cascade R-CNN
    # - We pass num_classes to automatically create the FastRCNNPredictor heads
    #   for the specific class count (replacing the default heads).
    # - We inject the specific RPN and ROI hyperparameters required for the task.
    model = CascadeRCNN(
        backbone,
        num_classes=num_classes,
        # RPN: Keep 2000 proposals after NMS during testing (default is 1000)
        rpn_post_nms_top_n_test=2000,
        # ROI Heads: Allow up to 1200 detections per image (default is 100)
        box_detections_per_img=1200,
        # ROI Heads: Set score threshold to 0.35 (default is 0.05)
        box_score_thresh=0.35,
    )

    return model
