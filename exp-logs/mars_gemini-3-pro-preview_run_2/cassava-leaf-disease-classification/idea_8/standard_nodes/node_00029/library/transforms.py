import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup

from library.config import Config


def get_transforms(stage: str, image_size: int):
    """
    Constructs the data transformation pipeline for the specified stage.

    Args:
        stage (str): The stage of the pipeline ('train', 'valid', 'test').
        image_size (int): The target resolution for the images (e.g., 224, 384).

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    # Standard ImageNet normalization statistics
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if stage == "train":
        # Training pipeline:
        # 1. RandomResizedCrop: Forces learning of scale-invariant features.
        # 2. RandomHorizontalFlip: Basic augmentation.
        # 3. RandAugment: Automated photometric distortions.
        # 4. ToTensor & Normalize.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    elif stage in ["valid", "test"]:
        # Validation/Inference pipeline:
        # 1. Resize: Resize the shorter edge to slightly larger than target (crop_pct=0.875).
        # 2. CenterCrop: Crop the center to the target image_size.
        # 3. ToTensor & Normalize.

        # Calculate resize dimension based on standard crop percentage for efficient validation
        crop_pct = 0.875
        resize_dim = int(image_size / crop_pct)

        return transforms.Compose(
            [
                transforms.Resize(
                    resize_dim, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    else:
        raise ValueError(
            f"Invalid stage: {stage}. Must be 'train', 'valid', or 'test'."
        )


def get_mixup_fn(phase_config):
    """
    Creates the Mixup/CutMix callable based on the training phase configuration.

    Args:
        phase_config (dict): A dictionary containing configuration for the current phase.
                             Expected to have 'mixup_prob'.

    Returns:
        timm.data.Mixup or None: The Mixup object if enabled, else None.
    """
    mixup_prob = phase_config.get("mixup_prob", 0.0)

    # If probability is 0 or less, we do not apply Mixup/CutMix
    if mixup_prob <= 0:
        return None

    # Initialize timm's Mixup implementation
    # It handles switching between Mixup and CutMix based on switch_prob (default 0.5)
    return Mixup(
        mixup_alpha=Config.mixup_alpha,
        cutmix_alpha=Config.cutmix_alpha,
        cutmix_minmax=None,  # Default
        prob=mixup_prob,  # Probability of applying mixup or cutmix
        switch_prob=0.5,  # Probability of switching to CutMix instead of Mixup
        mode="batch",  # Apply mixing per batch
        label_smoothing=Config.label_smoothing,
        num_classes=Config.num_classes,
    )
