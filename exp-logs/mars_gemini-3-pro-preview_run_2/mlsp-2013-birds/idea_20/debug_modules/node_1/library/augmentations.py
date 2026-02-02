import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline for the specified mode.

    Args:
        mode (str): One of "train", "val", "test".

    Returns:
        A.Compose: The composition of transforms.
    """
    # Config.IMG_SIZE is (Height, Width) -> (Frequency, Time)
    height, width = Config.IMG_SIZE

    if mode == "train":
        return A.Compose(
            [
                # Resize to unified resolution (Frequency x Time)
                A.Resize(height=height, width=width),
                # Photometric Distortions
                # Helps the model generalize to different recording conditions/gains
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # SpecAugment Simulation using CoarseDropout
                # Masks out blocks of time and frequency to force robust feature learning
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(height * 0.15),  # ~33 pixels
                    max_width=int(width * 0.15),  # ~67 pixels
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalize to ImageNet stats
                # Assumes input is converted to 3-channel RGB in the Dataset class
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                # Convert to PyTorch Tensor (C, H, W)
                ToTensorV2(),
            ]
        )

    elif mode in ["val", "test"]:
        return A.Compose(
            [
                # Deterministic Resize
                A.Resize(height=height, width=width),
                # Normalize to ImageNet stats
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                # Convert to PyTorch Tensor (C, H, W)
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'train', 'val', or 'test'.")
