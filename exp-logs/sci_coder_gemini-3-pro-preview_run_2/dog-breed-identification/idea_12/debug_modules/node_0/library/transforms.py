import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from library.config import Config


def get_interpolation_mode(mode_str):
    """
    Maps a string interpolation mode to the corresponding torchvision InterpolationMode.
    """
    mode_str = mode_str.lower()
    if mode_str == "bicubic":
        return InterpolationMode.BICUBIC
    elif mode_str == "bilinear":
        return InterpolationMode.BILINEAR
    elif mode_str == "nearest":
        return InterpolationMode.NEAREST
    elif mode_str == "lanczos":
        return InterpolationMode.LANCZOS
    else:
        # Default to Bicubic as it is generally safe for high-res models
        return InterpolationMode.BICUBIC


def get_stream_transforms(stream_config, is_train=False):
    """
    Generates the dictionary of transforms for a specific stream (A or B).

    This function constructs three distinct views for every image:
    1. Global: Resized to input_size x input_size (Squish). Preserves topology.
    2. Standard: Resized to view_standard_resize (aspect ratio preserved), then CenterCropped.
    3. Local: Resized to view_local_resize (Zoom in), then CenterCropped.

    Args:
        stream_config (dict): Configuration dictionary for the stream (e.g., Config.STREAM_A).
        is_train (bool): Flag indicating if this is for training.
                         Note: For this specific strategy (Feature Extraction -> LogReg),
                         transforms are deterministic even for 'train' split to ensure
                         consistent feature caching. TTA (flipping) is handled
                         externally during the extraction loop.

    Returns:
        dict: A dictionary containing 'global', 'standard', and 'local' transform pipelines.
    """

    # Extract configuration parameters
    mean = stream_config["mean"]
    std = stream_config["std"]
    interp_mode = get_interpolation_mode(stream_config.get("interpolation", "bicubic"))

    view_global_size = stream_config["view_global_size"]
    view_standard_resize = stream_config["view_standard_resize"]
    view_local_resize = stream_config["view_local_resize"]
    crop_size = stream_config["crop_size"]

    # Define common normalization transform
    # This handles the specific statistics for ConvNeXt (ImageNet) vs EVA02 (CLIP)
    normalize = transforms.Normalize(mean=mean, std=std)

    # -------------------------------------------------------------------------
    # 1. Global View (Squish)
    # -------------------------------------------------------------------------
    # Resizes the image to exactly (H, W), ignoring aspect ratio.
    # This ensures the model sees the entire object structure.
    global_transform = transforms.Compose(
        [
            transforms.Resize(
                (view_global_size, view_global_size), interpolation=interp_mode
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # -------------------------------------------------------------------------
    # 2. Standard View (Crop)
    # -------------------------------------------------------------------------
    # Resizes the smaller edge to 'view_standard_resize' while maintaining aspect ratio,
    # then takes a center crop. This matches standard pre-training protocols.
    standard_transform = transforms.Compose(
        [
            transforms.Resize(view_standard_resize, interpolation=interp_mode),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # -------------------------------------------------------------------------
    # 3. Local View (Zoom)
    # -------------------------------------------------------------------------
    # Resizes the smaller edge to a larger dimension 'view_local_resize', effectively
    # zooming in on the image, then takes a center crop. This forces the model
    # to focus on high-frequency texture details (fur, eyes, nose).
    local_transform = transforms.Compose(
        [
            transforms.Resize(view_local_resize, interpolation=interp_mode),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {
        "global": global_transform,
        "standard": standard_transform,
        "local": local_transform,
    }
