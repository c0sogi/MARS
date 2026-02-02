from torchvision import transforms
from library.config import CFG


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns the data augmentation and preprocessing pipeline for a given split.

    Args:
        split (str): The data split, one of 'train', 'val', or 'test'.

    Returns:
        transforms.Compose: A composition of transforms to be applied to the images.
    """
    # Standard ImageNet normalization statistics
    # These match the pre-trained weights of the ConvNeXt model
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    if split == "train":
        # Training pipeline:
        # 1. RandomResizedCrop: Forces the model to learn scale-invariant features
        #    and prevents overfitting to global image statistics.
        # 2. RandAugment: Applies a diverse set of photometric distortions (contrast,
        #    sharpness, etc.) automatically.
        # 3. ToTensor: Converts PIL image to Tensor (C, H, W) in [0, 1].
        # 4. Normalize: Standardizes input using ImageNet mean and std.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=CFG.image_size),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    elif split in ["val", "test"]:
        # Validation/Test pipeline:
        # 1. Resize: Resizes the shorter edge of the image to a slightly larger dimension
        #    (typically 256 for 224 input) to maintain aspect ratio.
        # 2. CenterCrop: Crops the center square of the target size.
        # 3. ToTensor & Normalize: Same as training.

        # Calculate resize dimension based on standard crop ratio (224/0.875 = 256)
        resize_dim = int(CFG.image_size / 0.875)

        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(CFG.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")
