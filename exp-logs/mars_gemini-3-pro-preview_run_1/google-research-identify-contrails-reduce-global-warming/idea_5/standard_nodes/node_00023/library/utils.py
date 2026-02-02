import numpy as np
from library.config import Config


def normalize_range(data, min_val, max_val):
    """
    Normalizes data to [0, 1] range using provided min and max values.
    Clips values outside the range.

    Args:
        data: Numpy array of input data.
        min_val: Minimum value for scaling (maps to 0).
        max_val: Maximum value for scaling (maps to 1).

    Returns:
        Numpy array normalized and clipped to [0, 1].
    """
    # Avoid division by zero if min == max (though unlikely with fixed config)
    if max_val == min_val:
        return np.zeros_like(data)

    data = (data - min_val) / (max_val - min_val)
    return np.clip(data, 0, 1)


def ash_composite(band_11, band_14, band_15):
    """
    Creates a false color Ash composite from GOES-16 bands.

    The composite is based on physical properties of contrails:
    - Red: Optical Depth (Band 15 - Band 14)
    - Green: Particle Phase (Band 14 - Band 11)
    - Blue: Temperature (Band 14)

    Args:
        band_11: Numpy array for Band 11 (8.4um) brightness temperatures.
        band_14: Numpy array for Band 14 (11.2um) brightness temperatures.
        band_15: Numpy array for Band 15 (12.3um) brightness temperatures.

    Returns:
        Numpy array of shape (..., 3) with normalized Ash composite.
    """
    # Red: Optical Depth proxy (Band 15 - Band 14)
    r = normalize_range(band_15 - band_14, Config.ASH_RED_MIN, Config.ASH_RED_MAX)

    # Green: Particle Phase proxy (Band 14 - Band 11)
    g = normalize_range(band_14 - band_11, Config.ASH_GREEN_MIN, Config.ASH_GREEN_MAX)

    # Blue: Temperature (Band 14)
    b = normalize_range(band_14, Config.ASH_BLUE_MIN, Config.ASH_BLUE_MAX)

    # Stack along the last dimension to create (H, W, 3) or (H, W, T, 3)
    return np.stack([r, g, b], axis=-1)


def rle_encode(mask):
    """
    Run-Length Encode a binary mask.

    The metric expects pixels numbered from top to bottom, then left to right.

    Args:
        mask: Binary mask of shape (H, W). 1 indicates mask, 0 background.

    Returns:
        String containing pairs of values 'start length'.
        Returns '-' if the mask is empty.
    """
    # Flatten column-major (Fortran style) as per competition requirement
    pixels = mask.flatten(order="F")

    # Check for empty mask
    if np.sum(pixels) == 0:
        return "-"

    # Pad with zeros at start and end to detect all transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first run (1-based index)
    # runs[1] is the start of the first gap
    # The length of the run is runs[1] - runs[0]
    # We update the even indices (lengths) by subtracting the odd indices (starts)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)
