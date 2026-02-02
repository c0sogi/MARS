import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from library.config import Config


def get_model(
    num_classes=Config.NUM_CLASSES,
    min_size=Config.MIN_SIZE,
    max_size=Config.MAX_SIZE,
    rpn_post_nms_top_n_test=Config.RPN_POST_NMS_TOP_N_TEST,
    detections_per_img=Config.DETECTIONS_PER_IMG,
    score_thresh=Config.SCORE_THRESH,
    nms_thresh=Config.NMS_THRESH,
):
    """
    Constructs the Faster R-CNN model with ResNet-50-FPN backbone.
    Configured for high-density text detection (Kuzushiji).

    Args:
        num_classes (int): Number of classes including background.
        min_size (int): Minimum size of the image to be rescaled to.
        max_size (int): Maximum size of the image to be rescaled to.
        rpn_post_nms_top_n_test (int): Number of proposals to keep after NMS during testing.
        detections_per_img (int): Maximum number of detections per image.
        score_thresh (float): Score threshold for predictions.
        nms_thresh (float): NMS threshold for predictions.
    """
    # Load the model with default pre-trained weights (ResNet-50-FPN on COCO)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # 1. Replace the Classifier Head
    # The pre-trained head has 91 classes (COCO). We need to replace it with one
    # that matches our specific number of Kuzushiji characters (plus background).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Configure RPN (Region Proposal Network)
    # Increase post-NMS proposals to capture all characters in dense pages.
    # Standard 1000 is often insufficient for pages with ~600 characters.
    model.rpn.post_nms_top_n_test = rpn_post_nms_top_n_test

    # 3. Configure RoI Heads (Box Predictor)
    # Increase the maximum number of detections per image.
    # The default of 100 is a hard ceiling that limits recall on this dataset.
    model.roi_heads.detections_per_img = detections_per_img

    # Set confidence and NMS thresholds
    # score_thresh filters low-confidence boxes
    # nms_thresh handles overlapping boxes
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = nms_thresh

    # 4. Configure Transform (Input Resizing)
    # Ensure the model's internal resizing matches the high-resolution input
    # provided by the dataset. This prevents the model from downscaling images
    # back to the default 800px, which would result in loss of detail for small characters.
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    return model
