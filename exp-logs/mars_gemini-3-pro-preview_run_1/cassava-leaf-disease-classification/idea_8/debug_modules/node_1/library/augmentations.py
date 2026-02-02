import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_train_transforms(image_size: int = None):
    """
    Constructs the training image transformation pipeline.

    Args:
        image_size (int, optional): The target height and width for the image.
                                    If None, defaults to Config.p1_image_size.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    if image_size is None:
        image_size = Config.p1_image_size

    return A.Compose(
        [
            # Contextual Cropping: Constrain scale to avoid ambiguous macro-crops
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=Config.random_crop_scale
            ),
            # Geometric Augmentations
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            # Color Augmentations
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            # Normalization and Tensor Conversion
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms(image_size: int = None):
    """
    Constructs the validation/inference image transformation pipeline.

    Args:
        image_size (int, optional): The target height and width for the image.
                                    If None, defaults to Config.p1_image_size.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    if image_size is None:
        image_size = Config.p1_image_size

    return A.Compose(
        [
            # Deterministic Resize
            A.Resize(height=image_size, width=image_size),
            # Normalization and Tensor Conversion
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )
