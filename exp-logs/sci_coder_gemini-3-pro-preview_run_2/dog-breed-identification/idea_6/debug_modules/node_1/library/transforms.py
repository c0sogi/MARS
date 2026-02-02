import torchvision.transforms as T
from library.config import Config


def get_stream_transforms(stream_config):
    """
    Constructs the Multi-View data transformation pipelines for a given stream configuration.

    This factory function generates three distinct geometric views for each image:
    1. Standard: Resize (maintain aspect ratio) -> Center Crop. Balanced view.
    2. Global: Resize (squish) to square. Preserves context/shape at cost of distortion.
    3. Local: Zoom (Resize large) -> Center Crop. Preserves texture/detail at cost of context.

    Args:
        stream_config (dict): The configuration dictionary for the specific stream
                              (e.g., Config.STREAM_A or Config.STREAM_B).
                              Must contain the 'input_size' key.

    Returns:
        dict: A dictionary with keys 'standard', 'global', 'local', where each value
              is a torchvision.transforms.Compose pipeline.
    """

    # Extract model-specific input resolution
    input_size = stream_config["input_size"]

    # Standard ImageNet normalization statistics
    # Used by both ConvNeXt (V1) and ViT (SWAG) recipes
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Bicubic interpolation is the standard for modern high-performance models
    interpolation = T.InterpolationMode.BICUBIC

    # Common preprocessing steps (Tensor conversion + Normalization)
    common_transforms = [T.ToTensor(), T.Normalize(mean=mean, std=std)]

    transforms = {}

    # --- 1. Standard View ---
    # Resizes the shortest edge to be slightly larger than the input size (approx 1.14x),
    # then performs a center crop. This is the standard evaluation protocol.
    # Calculation: input_size / 0.875 is the inverse of the standard crop fraction.
    resize_size_standard = int(input_size / 0.875)

    transforms["standard"] = T.Compose(
        [
            T.Resize(resize_size_standard, interpolation=interpolation),
            T.CenterCrop(input_size),
            *common_transforms,
        ]
    )

    # --- 2. Global View ---
    # "Squishes" the image to the exact input dimensions, ignoring aspect ratio.
    # This ensures the model sees the entire object structure (e.g., legs + head)
    # even if the object is long or wide.
    transforms["global"] = T.Compose(
        [
            T.Resize((input_size, input_size), interpolation=interpolation),
            *common_transforms,
        ]
    )

    # --- 3. Local View ---
    # Simulates a "Zoom" operation by resizing the image to a much larger scale
    # and then cropping the center. This feeds the model high-frequency texture details.
    resize_size_local = int(input_size * Config.LOCAL_VIEW_SCALE)

    transforms["local"] = T.Compose(
        [
            T.Resize(resize_size_local, interpolation=interpolation),
            T.CenterCrop(input_size),
            *common_transforms,
        ]
    )

    return transforms
