import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def get_model(num_classes):
    """
    Initializes a Faster R-CNN model with a ResNet-50-FPN backbone.
    The head is modified to support the specific number of classes for the task.

    Args:
        num_classes (int): The total number of classes including the background.
                           For this task, it is 4 (0=Background, 1=Typical,
                           2=Indeterminate, 3=Atypical).

    Returns:
        model (torch.nn.Module): The configured Faster R-CNN model.
    """
    # Load the pre-trained model using the default weights (COCO V1)
    # This provides a strong feature extractor backbone.
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    # Increase RPN proposal capacity and max detections for dense scenes
    # Cite solution_lesson_node_00005
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, rpn_post_nms_top_n_train=3000, box_detections_per_img=200
    )

    # Get the number of input features for the classifier head.
    # This is required to connect the new predictor to the existing backbone.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained box predictor with a new one.
    # FastRCNNPredictor creates two heads:
    # 1. Classification head: outputs scores for 'num_classes'
    # 2. Regression head: outputs 4 coordinates per class (or shared)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
