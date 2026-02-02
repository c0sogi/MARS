import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from library.config import Config, seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


def get_detection_model(num_classes):
    """
    Creates a Faster R-CNN model with a ResNet50-FPN backbone.
    Cite solution_lesson_node_00001: Using Object Detection to regress bounding boxes.
    Cite solution_lesson_node_00005: Increasing input resolution.
    """
    # Load pre-trained model
    # Weights=DEFAULT loads the best available weights (usually COCO)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier head
    # num_classes includes background (so actual classes + 1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Update transform parameters for higher resolution
    model.transform.min_size = (Config.DET_MIN_SIZE,)
    model.transform.max_size = Config.DET_MAX_SIZE

    return model
