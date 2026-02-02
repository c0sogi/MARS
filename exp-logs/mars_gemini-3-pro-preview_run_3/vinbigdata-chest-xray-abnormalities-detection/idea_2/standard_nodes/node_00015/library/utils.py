import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class Averager:
    """
    Computes and stores the average and current value of a metric/loss.
    """

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        return self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    """
    Custom collate function for object detection.
    Torchvision detection models expect a tuple of images (list of tensors)
    and targets (list of dicts), rather than a stacked tensor.
    """
    return tuple(zip(*batch))


def get_transforms(train=True):
    """
    Returns the data augmentation pipeline using Albumentations.

    Args:
        train (bool): If True, returns the training pipeline with augmentations.
                      If False, returns the validation/test pipeline (resize/norm only).
    """
    if train:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


def format_prediction_string(boxes, scores, labels):
    """
    Formats the raw model predictions into the submission CSV string format.

    Args:
        boxes (tensor/array): Bounding boxes in [xmin, ymin, xmax, ymax] format.
        scores (tensor/array): Confidence scores.
        labels (tensor/array): Model class IDs.

    Returns:
        str: A string formatted as "class_id confidence xmin ymin xmax ymax ..."
             or the "No finding" string if no objects are detected.
    """
    # Ensure inputs are numpy arrays
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Filter detections by confidence threshold
    valid_indices = np.where(scores > Config.CONFIDENCE_THRESHOLD)[0]

    # If no valid detections, return the "No finding" string
    if len(valid_indices) == 0:
        return Config.NO_FINDING_STRING

    pred_strings = []
    for i in valid_indices:
        class_id_model = int(labels[i])

        # Skip background class (0) if it appears
        if class_id_model == 0:
            continue

        score = float(scores[i])
        box = boxes[i]

        # Map model class ID (1-15) back to dataset class ID (0-13)
        if class_id_model in Config.MODEL_TO_DATASET_MAPPING:
            dataset_id = Config.MODEL_TO_DATASET_MAPPING[class_id_model]
            xmin, ymin, xmax, ymax = box

            # Format: class score xmin ymin xmax ymax
            # Using 4 decimal places for score and 1 for coordinates for precision
            pred_strings.append(
                f"{dataset_id} {score:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )

    # Double check if any strings were added (e.g. if all were background)
    if not pred_strings:
        return Config.NO_FINDING_STRING

    return " ".join(pred_strings)
