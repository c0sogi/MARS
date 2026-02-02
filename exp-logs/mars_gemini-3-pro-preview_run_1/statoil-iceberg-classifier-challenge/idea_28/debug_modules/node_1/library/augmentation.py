import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_training_transforms(
    img_size=Config.IMG_SIZE,
    hflip_prob=Config.AUG_HFLIP_PROB,
    vflip_prob=Config.AUG_VFLIP_PROB,
    rotate90_prob=Config.AUG_ROTATE_90_PROB,
    shift_scale_rotate_prob=Config.AUG_SHIFT_SCALE_ROTATE_PROB,
    shift_limit=Config.AUG_SHIFT_LIMIT,
    scale_limit=Config.AUG_SCALE_LIMIT,
    rotate_limit=Config.AUG_ROTATE_LIMIT,
):
    """
    Returns the composition of image transformations for the training set.
    Includes geometric augmentations and resizing with Bicubic interpolation.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=hflip_prob),
            A.VerticalFlip(p=vflip_prob),
            A.RandomRotate90(p=rotate90_prob),
            A.ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                border_mode=cv2.BORDER_REFLECT,
                value=0,
                mask_value=0,
                p=shift_scale_rotate_prob,
            ),
            ToTensorV2(),
        ]
    )


def get_validation_transforms(img_size=Config.IMG_SIZE):
    """
    Returns the composition of image transformations for validation and testing.
    Performs deterministic resizing with Bicubic interpolation and tensor conversion.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
            ToTensorV2(),
        ]
    )


def get_test_transforms(img_size=Config.IMG_SIZE):
    """
    Alias for validation transforms, used for inference on the test set.
    """
    return get_validation_transforms(img_size=img_size)
