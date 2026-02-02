import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from library.config import Config, seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


def get_detection_model(num_classes):
    """
    Returns a Faster R-CNN model with a ResNet50-FPN backbone.
    """
    # Load model pre-trained on COCO
    # Using default weights (IMAGENET/COCO)
    # Cite solution_lesson_node_00005: Preserving Spatial Fidelity in Document Object Detection via Resolution Scaling
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=1024, max_size=2048)

    # Replace the classifier with a new one that has num_classes
    # num_classes includes the background class (so actual classes + 1)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
