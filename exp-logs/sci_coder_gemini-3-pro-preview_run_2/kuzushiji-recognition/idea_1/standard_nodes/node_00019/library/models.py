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
    # Cite solution_lesson_node_00008: Increasing box_detections_per_img to handle dense pages
    # Cite solution_lesson_node_00015: Passing resize params to constructor to ensure they take effect
    # Cite solution_lesson_node_00016: Increasing RPN proposals to uncap recall on dense pages
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        box_detections_per_img=1000,
        min_size=Config.DET_MIN_SIZE,
        max_size=Config.DET_MAX_SIZE,
        rpn_pre_nms_top_n_test=Config.RPN_PRE_NMS_TOP_N_TEST,
        rpn_post_nms_top_n_test=Config.RPN_POST_NMS_TOP_N_TEST,
    )

    # Replace the classifier head
    # num_classes includes background (so actual classes + 1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
