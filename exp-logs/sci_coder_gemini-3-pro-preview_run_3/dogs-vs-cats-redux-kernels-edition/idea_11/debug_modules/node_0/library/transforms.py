import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transforms(image_size: int, mode: str = "train") -> transforms.Compose:
    """
    Returns the data transformation pipeline based on the image size and mode.

    Args:
        image_size (int): The target resolution for the images (e.g., 224, 256).
        mode (str): The operation mode, either 'train', 'valid', or 'test'.

    Returns:
        transforms.Compose: Composed torchvision transforms.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Context-Preserving Augmentation:
                # Scale (0.8, 1.0) ensures we don't crop out too much of the subject
                # Bicubic interpolation is used for higher quality resizing
                transforms.RandomResizedCrop(
                    (image_size, image_size),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                # Color Jitter with >= 0.2 intensity to handle lighting variance
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test:
        # Deterministic resize to the target resolution.
        # We use direct resizing to (image_size, image_size) to ensure the model
        # sees the entire image context, avoiding center-crop information loss.
        return transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
