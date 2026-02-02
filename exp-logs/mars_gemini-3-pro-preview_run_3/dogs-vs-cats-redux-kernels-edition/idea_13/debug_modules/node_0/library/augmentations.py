import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(img_size: int, split: str):
    """
    Constructs the data augmentation pipeline based on the split and image size.

    Implements the 'Decoupled Input Pipeline' strategy:
    - Enforces Bicubic interpolation for all resizing operations.
    - Applies context-preserving RandomResizedCrop for training.
    - Applies ColorJitter for lighting invariance.

    Args:
        img_size (int): The target spatial dimension (e.g., 224 or 256).
        split (str): The data split ('train', 'val', or 'test').

    Returns:
        A.Compose: An albumentations composition of transforms.
    """

    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        transforms = [
            # Context-Preserving Augmentation
            # Scale (0.8, 1.0) prevents aggressive cropping of the subject.
            # Bicubic interpolation is used to align with modern pre-training recipes.
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=Config.AUG_CROP_SCALE,
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            # Geometric Augmentation
            A.HorizontalFlip(p=0.5),
            # Photometric Augmentation
            # Intensity set via Config (>= 0.2)
            A.ColorJitter(
                brightness=Config.AUG_COLOR_JITTER,
                contrast=Config.AUG_COLOR_JITTER,
                saturation=Config.AUG_COLOR_JITTER,
                hue=Config.AUG_COLOR_JITTER,
                p=0.5,
            ),
            # Normalization & Conversion
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    else:
        # Validation / Test Pipeline
        # Deterministic resizing with Bicubic interpolation
        transforms = [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]

    return A.Compose(transforms)
