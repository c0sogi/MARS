import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from library.config import Config


def get_model(num_classes, config=None):
    """
    Constructs the object detection model for Kuzushiji Character Recognition.

    Uses a Faster R-CNN with ResNet-50-FPN backbone, tuned with specific
    hyperparameters to handle dense, small characters (High-Res, High-Recall).

    Args:
        num_classes (int): Total number of classes (characters + background).
        config (Config, optional): Configuration object containing hyperparameters.

    Returns:
        model (torch.nn.Module): The configured PyTorch model.
    """
    if config is None:
        config = Config()

    # Load the model with default pre-trained weights (ResNet-50-FPN on COCO)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # -------------------------------------------------------------------------
    # 1. Replace Classification Head
    # -------------------------------------------------------------------------
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one for our specific number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # -------------------------------------------------------------------------
    # 2. Configure Input Transform (Resolution Scaling)
    # -------------------------------------------------------------------------
    # High resolution is non-negotiable for resolving small characters.
    # We enforce a minimum size of 1024px and a maximum of 2048px.
    model.transform.min_size = (config.MIN_SIZE,)
    model.transform.max_size = config.MAX_SIZE

    # -------------------------------------------------------------------------
    # 3. Configure Region Proposal Network (RPN)
    # -------------------------------------------------------------------------
    # Standard RPN keeps 1000 proposals. For pages with ~600 characters,
    # we need more candidates to ensure high recall.
    model.rpn.post_nms_top_n_test = config.RPN_POST_NMS_TOP_N_TEST

    # -------------------------------------------------------------------------
    # 4. Configure ROI Heads (Detections)
    # -------------------------------------------------------------------------
    # Standard Faster R-CNN caps detections at 100 per image.
    # We override this to 1200 to match the dataset density.
    model.roi_heads.detections_per_img = config.BOX_DETECTIONS_PER_IMG

    # Set the inference score threshold.
    # A lower threshold (0.35) is used to prioritize recall for faint ink.
    model.roi_heads.score_thresh = config.SCORE_THRESHOLD

    return model
