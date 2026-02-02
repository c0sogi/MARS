import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=380):
    """
    Returns the training transformations pipeline.

    Args:
        image_size (int): The target resolution for resizing. Defaults to 380.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Cite Lesson 13: Prefer CoarseDropout over CutMix for localized features.
            # Cite Lesson 6: Scale occlusion size (72x72).
            # Cite Lesson 5: Enforce minimum dimensions (8x8).
            A.CoarseDropout(
                max_holes=8,
                max_height=72,
                max_width=72,
                min_holes=None,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.5,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(image_size=380):
    """
    Returns the validation/test transformations pipeline.

    Args:
        image_size (int): The target resolution for resizing. Defaults to 380.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
