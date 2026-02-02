import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from library.config import Config


def get_model(num_classes=Config.NUM_CLASSES):
    """
    Initializes the Faster R-CNN model with a ResNet-50-FPN backbone.

    This function loads a pre-trained model (trained on COCO train2017) and replaces
    the box predictor head (classification and regression) to match the target
    number of classes for the VinBigData task.

    Args:
        num_classes (int): The total number of classes including the background.
                           Defaults to Config.NUM_CLASSES (15).

    Returns:
        torch.nn.Module: The modified Faster R-CNN model ready for training or inference.
    """
    # Load the model with the best available pre-trained weights (typically COCO)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier in the box predictor
    # This is required to ensure the new head matches the backbone's output
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained box predictor with a new one
    # FastRCNNPredictor creates two fully connected layers:
    # 1. Classification scores (num_classes)
    # 2. Bounding box regression deltas (num_classes * 4)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
