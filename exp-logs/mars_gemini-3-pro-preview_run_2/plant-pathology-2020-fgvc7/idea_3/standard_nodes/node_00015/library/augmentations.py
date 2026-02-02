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
            # Cite solution_lesson_node_00004: Aggressive Information Dropping
            # Cite solution_lesson_node_00006: Scaling Cutout Occlusion
            # Cite solution_lesson_node_00005: Enforcing Minimum Dimensions
            A.CoarseDropout(
                max_holes=8,
                max_height=int(image_size * 0.2),
                max_width=int(image_size * 0.2),
                min_holes=4,
                min_height=int(image_size * 0.05),
                min_width=int(image_size * 0.05),
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
        image_size (int): The target resolution for resizing. Defaults to 480.

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
