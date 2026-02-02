import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import IMG_SIZE


def get_train_transforms(img_size=IMG_SIZE):
    """
    Returns the training image augmentation pipeline.

    Strategy:
    - Resize to fixed input size.
    - Conservative Geometric Augmentations:
        - Horizontal Flip (p=0.5)
        - Rotation +/- 20 degrees
        - Scale +/- 10% (0.9 - 1.1)
    - Photometric Augmentations:
        - Brightness and Contrast only.
        - EXPLICITLY EXCLUDING Saturation and Hue to avoid chromatic noise.
    - Normalization (ImageNet stats).

    Args:
        img_size (int): The target height and width of the image.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            # Geometric Augmentations
            A.HorizontalFlip(p=0.5),
            # Combined Affine transform for efficiency
            # border_mode=cv2.BORDER_CONSTANT (0) fills new pixels with black (0)
            # This prevents reflection artifacts which can be confusing for identification
            A.ShiftScaleRotate(
                shift_limit=0.0,  # No translation
                scale_limit=0.1,  # 0.9 to 1.1
                rotate_limit=15,  # +/- 15 degrees (Cite solution_lesson_node_00014)
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            # Photometric Augmentations
            # Strictly limiting to Brightness and Contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # Normalization and Tensor Conversion
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_test_transforms(img_size=IMG_SIZE):
    """
    Returns the validation/test image augmentation pipeline.

    Strategy:
    - Deterministic Resize.
    - Normalization (ImageNet stats).

    Args:
        img_size (int): The target height and width of the image.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
