import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the data augmentation and preprocessing pipeline for the specified phase.

    Implements Letterbox Resizing:
    1. Resize the longest dimension to Config.IMG_SIZE while maintaining aspect ratio.
    2. Pad the shorter dimension with zeros to create a square image of size Config.IMG_SIZE.

    Args:
        phase (str): One of 'train', 'val', or 'test'.

    Returns:
        A.Compose: The albumentations transform pipeline.
    """

    # Base transforms applied to all phases (Resizing logic)
    # LongestMaxSize: Resizes image so that the longest side is equal to Config.IMG_SIZE
    # PadIfNeeded: Pads the remaining side to match Config.IMG_SIZE, resulting in a square image
    base_transforms = [
        A.LongestMaxSize(max_size=Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,  # Pad with black
            mask_value=0,
        ),
    ]

    if phase == "train":
        # Augmentations for training
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # Optional: ShiftScaleRotate can be helpful, but sticking to prompt specifics
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5)
        ]
    else:
        # No augmentations for val/test
        augmentations = []

    # Normalization and Tensor conversion
    # Using standard ImageNet mean and std
    final_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    # Compose the pipeline
    # Bounding box parameters are required for detection tasks
    # We use 'pascal_voc' format: [x_min, y_min, x_max, y_max]
    return A.Compose(
        base_transforms + augmentations + final_transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )
