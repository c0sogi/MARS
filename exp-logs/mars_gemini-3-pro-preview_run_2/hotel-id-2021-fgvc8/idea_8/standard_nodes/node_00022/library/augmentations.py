import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(image_size: int, mode: str = "train"):
    """
    Returns the image transformation pipeline based on the mode and target image size.

    Args:
        image_size (int): The target height and width of the image (e.g., 256 or 384).
        mode (str): The pipeline mode. Options: 'train', 'valid', 'test'.

    Returns:
        A.Compose: The albumentations composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentations
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                # Regularization (Cutout-like)
                # Scaling hole sizes relative to image size for consistency across resolutions
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(image_size * 0.1),
                    max_width=int(image_size * 0.1),
                    min_holes=1,
                    min_height=int(image_size * 0.05),
                    min_width=int(image_size * 0.05),
                    fill_value=0,
                    p=0.2,
                ),
                # Normalization and Tensor Conversion
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif mode in ["valid", "test"]:
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

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported modes are 'train', 'valid', 'test'."
        )
