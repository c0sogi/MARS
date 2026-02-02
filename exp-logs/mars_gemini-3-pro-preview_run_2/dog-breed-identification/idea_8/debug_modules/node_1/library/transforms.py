import torch
from torchvision import transforms
import library.config as config


def get_global_transform():
    """
    Generates the transform for the Global View.

    Strategy:
    - Resize to (224, 224) explicitly. This 'squishes' the image if the aspect ratio
      is not square, ensuring the entire object is visible (no cropping).
    - Convert to Tensor.
    - Normalize using ImageNet statistics.
    """
    return transforms.Compose(
        [
            transforms.Resize(config.GLOBAL_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ]
    )


def get_standard_transform():
    """
    Generates the transform for the Standard View.

    Strategy:
    - Resize the smaller edge to 232 (maintaining aspect ratio).
    - Center Crop to 224x224.
    - This matches the standard pre-training evaluation recipe.
    """
    return transforms.Compose(
        [
            transforms.Resize(config.STANDARD_RESIZE),
            transforms.CenterCrop(config.STANDARD_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ]
    )


def get_local_transform():
    """
    Generates the transform for the Robust Local View.

    Strategy:
    - Resize the smaller edge to 288 (zoomed in relative to 224 crop).
    - Generate 5 crops: Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center.
    - Convert each crop to Tensor and Normalize individually.
    - Stack into a single tensor of shape (5, C, H, W).
    """

    # Define internal helper to process the tuple of crops returned by FiveCrop
    def _process_five_crops(crops):
        # crops is a tuple of 5 PIL images
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=config.MEAN, std=config.STD)

        # Apply ToTensor and Normalize to each crop
        processed_crops = [normalize(to_tensor(crop)) for crop in crops]

        # Stack into a 4D tensor: (5, 3, 224, 224)
        return torch.stack(processed_crops)

    return transforms.Compose(
        [
            transforms.Resize(config.LOCAL_RESIZE),
            transforms.FiveCrop(config.LOCAL_CROP),
            transforms.Lambda(_process_five_crops),
        ]
    )
