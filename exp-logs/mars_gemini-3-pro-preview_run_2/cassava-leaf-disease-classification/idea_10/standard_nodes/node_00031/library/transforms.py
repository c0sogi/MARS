import math
from torchvision import transforms

# Standard ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(phase: str, img_size: int) -> transforms.Compose:
    """
    Constructs the data transformation pipeline for a specific phase and image resolution.

    Args:
        phase (str): The execution phase. Options: 'train', 'valid', 'inference'.
        img_size (int): The target spatial resolution (e.g., 224, 384).

    Returns:
        transforms.Compose: A composition of torchvision transforms.
    """
    phase = phase.lower()

    if phase == "train":
        # Training Pipeline: Heavy augmentation for regularization
        return transforms.Compose(
            [
                # Geometric Augmentations
                # RandomResizedCrop forces the model to learn features at different scales
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                # Photometric Augmentations
                # RandAugment applies a sequence of random distortions (e.g., shear, contrast)
                # Standard settings: num_ops=2, magnitude=9
                transforms.RandAugment(num_ops=2, magnitude=9),
                # Conversion and Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    elif phase in ["valid", "val", "validation", "test", "inference"]:
        # Validation/Inference Pipeline: Deterministic preprocessing
        # We resize the image to be slightly larger than the target size and take a center crop.
        # This ratio (256/224 = 1.142 -> 1/0.875) is standard for classification evaluation.
        crop_pct = 0.875
        resize_size = int(math.floor(img_size / crop_pct))

        return transforms.Compose(
            [
                # Resize the smaller edge to resize_size, maintaining aspect ratio
                transforms.Resize(
                    resize_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(img_size),
                # Conversion and Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    else:
        raise ValueError(
            f"Unknown phase: {phase}. Expected 'train', 'valid', or 'inference'."
        )
