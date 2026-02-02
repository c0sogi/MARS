import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from library.config import Config, seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


def get_detection_model(num_classes):
    """
    Returns a Faster R-CNN model with ResNet50-FPN backbone.
    Cite solution_lesson_node_00001: Using Object Detection architecture.
    """
    # Load pre-trained model
    # Cite solution_lesson_node_00005: Increasing input resolution parameters via constructor
    try:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=Config.MIN_SIZE,
            max_size=Config.MAX_SIZE,
            box_detections_per_img=Config.BOX_DETECTIONS_PER_IMG,
        )
    except:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            min_size=Config.MIN_SIZE,
            max_size=Config.MAX_SIZE,
            box_detections_per_img=Config.BOX_DETECTIONS_PER_IMG,
        )

    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
