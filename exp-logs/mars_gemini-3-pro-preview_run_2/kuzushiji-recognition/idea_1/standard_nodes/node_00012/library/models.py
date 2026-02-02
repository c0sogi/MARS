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
    Cite solution_lesson_node_00001: Using Object Detection (Faster R-CNN) to handle dense instances.
    """
    # Load model pre-trained on COCO
    # Using default weights (IMAGENET/COCO)
    # Increase min_size/max_size to preserve details on large pages
    # Cite solution_lesson_node_00008: Increase box_detections_per_img to 1000 to handle dense pages.
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=1024, max_size=2048, box_detections_per_img=1000
    )

    # Replace the classifier with a new one that has num_classes
    # num_classes includes the background class (so actual classes + 1)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
