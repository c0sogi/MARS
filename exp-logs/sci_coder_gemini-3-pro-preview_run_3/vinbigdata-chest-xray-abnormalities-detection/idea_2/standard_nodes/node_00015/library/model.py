import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    """
    Initializes a Faster R-CNN model with a ResNet50-FPN backbone.

    The model is pre-trained on COCO. The box predictor head is replaced
    to match the number of classes in the target dataset.

    Args:
        num_classes (int): Total number of classes including the background.
                           For VinDr-CXR: 14 findings + 1 background = 15.

    Returns:
        model (torch.nn.Module): The configured Faster R-CNN model.
    """
    # Load the model with default pre-trained weights (COCO)
    # This provides a strong feature extractor for transfer learning
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    # This is usually 1024 for ResNet50-FPN
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one (randomly initialized)
    # configured for the specific number of classes in our dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
