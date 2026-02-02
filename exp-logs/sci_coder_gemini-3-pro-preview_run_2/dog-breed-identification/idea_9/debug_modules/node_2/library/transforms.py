import torch
from torchvision import transforms
from library import config


def get_view_transforms():
    """
    Constructs and returns the image transformation pipelines for the Multi-Scale
    Deep Feature Pyramid strategy.

    Returns:
        dict: A dictionary containing 'global', 'standard', and 'local' transformation pipelines.
    """

    # Standard ImageNet normalization
    # Applied as the final step in all pipelines
    normalize = transforms.Normalize(mean=config.MEAN, std=config.STD)

    # -------------------------------------------------------------------------
    # View 1: Global View (Shape)
    # -------------------------------------------------------------------------
    # Resizes the image to a fixed square size (Squish).
    # This preserves the entire field of view, capturing global shape information
    # at the cost of aspect ratio distortion.
    global_transform = transforms.Compose(
        [
            transforms.Resize((config.GLOBAL_VIEW_SIZE, config.GLOBAL_VIEW_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # -------------------------------------------------------------------------
    # View 2: Standard View (Context)
    # -------------------------------------------------------------------------
    # Traditional resize-then-crop approach.
    # Resizes the smaller edge to specific size (preserving aspect ratio)
    # and takes a center crop. Matches pre-training conditions.
    standard_transform = transforms.Compose(
        [
            transforms.Resize(config.STANDARD_VIEW_RESIZE),
            transforms.CenterCrop(config.STANDARD_VIEW_CROP),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # -------------------------------------------------------------------------
    # View 3: Robust Local View (Texture)
    # -------------------------------------------------------------------------
    # Resizes to a larger resolution and extracts 5 crops (4 corners + center).
    # This captures high-frequency texture details and mitigates localization errors.
    #
    # Note: FiveCrop returns a tuple of 5 PIL images. We use a Lambda transform
    # to convert each crop to a Tensor, normalize it, and stack them into a
    # single 4D Tensor of shape (5, C, H, W).
    local_transform = transforms.Compose(
        [
            transforms.Resize(config.LOCAL_VIEW_RESIZE),
            transforms.FiveCrop(config.LOCAL_VIEW_CROP),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [normalize(transforms.ToTensor()(crop)) for crop in crops]
                )
            ),
        ]
    )

    return {
        "global": global_transform,
        "standard": standard_transform,
        "local": local_transform,
    }
