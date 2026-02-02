import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import CFG


def get_transforms(data):
    """
    Returns the image transformation pipeline for the specified data split.

    Args:
        data (str): One of 'train', 'valid', or 'inference'.

    Returns:
        A.Compose: The albumentations transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                # Heavy Geometric Augmentations
                A.RandomResizedCrop(CFG.img_size, CFG.img_size),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Light Photometric Augmentations
                # Strictly limiting color-based augmentations (like HSV) to preserve disease signals.
                # Using mild brightness/contrast adjustments.
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.2
                ),
                # Normalization and Tensor Conversion
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                # Deterministic resizing for validation
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "inference":
        return A.Compose(
            [
                # Base transform for TTA (Resize + Normalize)
                # Flips for TTA are typically handled in the inference loop or dataset
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(
            f"Unknown data split: {data}. Expected 'train', 'valid', or 'inference'."
        )
