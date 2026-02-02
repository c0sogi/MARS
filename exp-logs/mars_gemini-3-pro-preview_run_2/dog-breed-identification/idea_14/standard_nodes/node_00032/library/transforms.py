import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import library.config as config


def get_stream_transforms(stream_type: str):
    """
    Returns a dictionary of transforms for the specified stream type.

    Args:
        stream_type (str): 'stream_a' (CNN) or 'stream_b' (ViT/DINOv2).

    Returns:
        dict: Keys are 'global', 'standard', 'local', values are torchvision.transforms.Compose.
    """
    # Determine interpolation mode based on stream type
    if stream_type == "stream_a":
        # Stream A: Supervised CNN (ConvNeXt) uses Bilinear interpolation
        interpolation = InterpolationMode.BILINEAR
    elif stream_type == "stream_b":
        # Stream B: Self-Supervised ViT (DINOv2) uses Bicubic interpolation
        interpolation = InterpolationMode.BICUBIC
    else:
        raise ValueError(
            f"Unknown stream_type: {stream_type}. Expected 'stream_a' or 'stream_b'."
        )

    # Common Normalization
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

    # -------------------------------------------------------------------------
    # View 1: Global
    # Strategy: Squish to (IMG_SIZE, IMG_SIZE). Preserves object topology but
    # distorts aspect ratio.
    # -------------------------------------------------------------------------
    global_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.IMG_SIZE, config.IMG_SIZE), interpolation=interpolation
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # -------------------------------------------------------------------------
    # View 2: Standard
    # Strategy: Resize smaller edge to RESIZE_STANDARD, then Center Crop.
    # Matches standard pre-training evaluation protocols.
    # -------------------------------------------------------------------------
    standard_transform = transforms.Compose(
        [
            transforms.Resize(config.RESIZE_STANDARD, interpolation=interpolation),
            transforms.CenterCrop(config.CROP_STANDARD),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # -------------------------------------------------------------------------
    # View 3: Local
    # Strategy: Resize smaller edge to RESIZE_LOCAL (Zoom), then Center Crop.
    # Focuses on fine-grained details by cropping a central region from a larger resize.
    # -------------------------------------------------------------------------
    local_transform = transforms.Compose(
        [
            transforms.Resize(config.RESIZE_LOCAL, interpolation=interpolation),
            transforms.CenterCrop(config.CROP_LOCAL),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {
        "global": global_transform,
        "standard": standard_transform,
        "local": local_transform,
    }
