import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import IMG_SIZE, PIXEL_MEAN, PIXEL_STD, seed_everything


def get_train_transforms():
    """
    Returns the training transformations including CLAHE, resizing, and augmentations.
    """
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_area=0, min_visibility=0
        ),
    )


def get_valid_transforms():
    """
    Returns the validation/test transformations (Resize + CLAHE + Normalize).
    """
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_area=0, min_visibility=0
        ),
    )


def collate_fn(batch):
    """
    Custom collate function for object detection.

    Args:
        batch: List of tuples (image, target, image_id) output by the Dataset.

    Returns:
        images: Stacked tensor of images (B, C, H, W).
        targets: List of target dictionaries (required for Faster R-CNN).
        image_ids: List of image ID strings.
    """
    images, targets, image_ids = tuple(zip(*batch))

    # Stack images into a single tensor
    images = torch.stack(images)

    # Targets should be a list of dictionaries for Faster R-CNN
    targets = list(targets)

    # Image IDs is a list of strings
    image_ids = list(image_ids)

    return images, targets, image_ids


def format_prediction_string(boxes, scores, labels):
    """
    Formats the image-level predictions into the submission string format.

    Args:
        boxes (list or np.array): List of bounding boxes [xmin, ymin, xmax, ymax].
        scores (list or np.array): List of confidence scores.
        labels (list or np.array): List of class labels (int).

    Returns:
        str: Formatted prediction string e.g., "opacity 0.5 100 100 200 200 ..."
             or "none 1 0 0 1 1" if no boxes.
    """
    # If no boxes predicted, return the "none" string
    if len(boxes) == 0:
        return "none 1 0 0 1 1"

    pred_strings = []
    for i, box in enumerate(boxes):
        score = scores[i]

        # As per requirements: map all granular classes to "opacity" for the image-level metric
        label_name = "opacity"

        xmin, ymin, xmax, ymax = box

        # Format: class_id confidence xmin ymin xmax ymax
        pred_strings.append(
            f"{label_name} {score:.6f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
        )

    return " ".join(pred_strings)


def format_study_prediction_string(label_name, score):
    """
    Formats the study-level prediction.

    Args:
        label_name (str): Full label name (e.g., 'Negative for Pneumonia', 'Typical Appearance').
        score (float): Confidence score.

    Returns:
        str: Formatted string e.g., "negative 0.9 0 0 1 1".
    """
    # Map full label names to submission IDs
    # 'Negative for Pneumonia' -> 'negative'
    # 'Typical Appearance' -> 'typical'
    # 'Indeterminate Appearance' -> 'indeterminate'
    # 'Atypical Appearance' -> 'atypical'

    short_id = label_name.split(" ")[0].lower()

    return f"{short_id} {score:.6f} 0 0 1 1"
