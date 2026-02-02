import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_train_transforms():
    """
    Returns the training data augmentation pipeline.

    Includes:
    - Resize to 256x256
    - Vertical and Horizontal Flips
    - Rotation (+/- 45 degrees)
    - Shift and Scale (strictly limited to +/- 5% scaling)
    - Normalization and Tensor conversion
    """
    return A.Compose(
        [
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            # Geometric Augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            # Constrained Scaling and Shifting
            # We separate rotation (handled above) from scaling to strictly control the scale limit
            # to +/- 5% as defined in Config, preventing the loss of small disease features.
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=Config.SHIFT_SCALE_ROTATE_LIMIT,
                rotate_limit=0,
                p=0.5,
            ),
            # Normalization (ImageNet defaults)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    """
    Returns the validation/test data augmentation pipeline.

    Includes:
    - Resize to 256x256
    - Normalization and Tensor conversion
    """
    return A.Compose(
        [
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
