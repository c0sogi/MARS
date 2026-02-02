import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import configuration
from library.config import Config


def cyclic_roll(image, shift_ratio=0.0):
    """
    Performs a cyclic shift of the spectrogram along the time axis (width).

    This is used for:
    1. Training Augmentation: Random shifts to enforce temporal invariance.
    2. Test-Time Augmentation (TTA): Deterministic shifts to stabilize predictions.

    Args:
        image (np.ndarray): Input image of shape (H, W, C).
        shift_ratio (float): The fraction of the width to shift.
                             Positive values shift right, negative left.
                             e.g., 0.5 shifts by half the width.

    Returns:
        np.ndarray: The rolled image.
    """
    if shift_ratio == 0.0:
        return image

    h, w, c = image.shape
    shift = int(w * shift_ratio)

    # np.roll shifts elements along the specified axis.
    # Axis 1 is the width (time) dimension.
    return np.roll(image, shift, axis=1)


def get_transforms(phase, model_name):
    """
    Returns the Albumentations composition for the specified phase and model.

    Adapts the input resolution based on the model architecture:
    - Anchors (ResNet18, EfficientNet-B0): 224x448
    - DenseNet121: 160x320

    Args:
        phase (str): 'train', 'val', or 'test'.
        model_name (str): Name of the model (e.g., 'resnet18', 'densenet121').

    Returns:
        A.Compose: The transform pipeline.
    """

    # Determine target size based on model type
    if model_name == "densenet121":
        height, width = Config.IMG_SIZE_DENSENET
    else:
        # Default to Anchor size for ResNet18 and EfficientNet-B0
        height, width = Config.IMG_SIZE_ANCHOR

    transforms_list = []

    # 1. Resize
    # We use simple interpolation.
    transforms_list.append(A.Resize(height=height, width=width, p=1.0))

    # 2. Training Augmentations (SpecAugment via CoarseDropout)
    if phase == "train":
        # Simulate SpecAugment (Time and Frequency masking) using CoarseDropout
        # We allow rectangular holes.
        # max_holes: Number of masked regions
        # max_height/max_width: Max size of the mask relative to image size
        transforms_list.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=int(height * 0.2),  # Mask up to 20% of freq
                max_width=int(width * 0.2),  # Mask up to 20% of time
                min_holes=2,
                min_height=int(height * 0.05),
                min_width=int(width * 0.05),
                fill_value=0,
                p=0.5,
            )
        )

    # 3. Normalization
    # Standard ImageNet normalization is used because we are using Pseudo-RGB
    # and pretrained backbones.
    transforms_list.append(
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        )
    )

    # 4. Convert to Tensor
    transforms_list.append(ToTensorV2(p=1.0))

    return A.Compose(transforms_list)
