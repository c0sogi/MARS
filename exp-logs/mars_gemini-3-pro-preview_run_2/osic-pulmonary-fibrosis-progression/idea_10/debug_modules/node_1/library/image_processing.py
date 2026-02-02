import numpy as np
import cv2
from library.config import Config


def get_patient_zones(volume):
    """
    Splits the 3D CT volume into N anatomical zones (e.g., Upper, Middle, Lower) along the Z-axis.

    Args:
        volume (np.ndarray): 3D array of shape (Depth, Height, Width).

    Returns:
        list of np.ndarray: A list containing the sub-volumes for each zone.
    """
    n_zones = Config.N_ZONES

    # Handle empty or insufficient volume
    if volume is None or volume.shape[0] == 0:
        # Return empty arrays for each zone
        return [np.zeros((0, 512, 512), dtype=np.float32) for _ in range(n_zones)]

    # If fewer slices than zones, we can't split strictly, so we split as best as possible
    # np.array_split handles cases where indices_or_sections > axis length
    zones = np.array_split(volume, n_zones, axis=0)

    return zones


def select_variance_slice(zone_volume):
    """
    Selects the single slice with the highest pixel variance from a zonal sub-volume.
    High variance often correlates with tissue heterogeneity (fibrosis/honeycombing).

    Args:
        zone_volume (np.ndarray): 3D array of shape (Depth, Height, Width).

    Returns:
        np.ndarray: 2D image of shape (Height, Width).
    """
    if zone_volume is None or zone_volume.shape[0] == 0:
        # Return a blank image if the zone is empty
        return np.zeros((512, 512), dtype=np.float32)

    max_var = -1.0
    best_slice = zone_volume[0]

    # Iterate through slices to find the one with maximum variance
    for i in range(zone_volume.shape[0]):
        current_slice = zone_volume[i]
        # Compute variance
        var = np.var(current_slice)

        if var > max_var:
            max_var = var
            best_slice = current_slice

    return best_slice


def compute_density_hist(zone_volume):
    """
    Computes a normalized histogram of Hounsfield Units (HU) for the zone.
    Bins correspond to clinical density ranges:
    1. <-950: Air / Severe Emphysema
    2. -950 to -700: Normal Lung
    3. -700 to -400: Ground Glass Opacity / Mild Fibrosis
    4. >-400: Consolidation / Fibrosis / Soft Tissue

    Args:
        zone_volume (np.ndarray): 3D array of shape (Depth, Height, Width).

    Returns:
        np.ndarray: 1D array of shape (4,) containing normalized counts.
    """
    if zone_volume is None or zone_volume.shape[0] == 0:
        return np.zeros(Config.DENSITY_BINS, dtype=np.float32)

    # Flatten the volume to 1D for histogram computation
    pixels = zone_volume.flatten()
    total_pixels = pixels.size

    if total_pixels == 0:
        return np.zeros(Config.DENSITY_BINS, dtype=np.float32)

    # Define thresholds
    t1 = -950
    t2 = -700
    t3 = -400

    # Compute counts
    c1 = np.sum(pixels < t1)
    c2 = np.sum((pixels >= t1) & (pixels < t2))
    c3 = np.sum((pixels >= t2) & (pixels < t3))
    c4 = np.sum(pixels >= t3)

    counts = np.array([c1, c2, c3, c4], dtype=np.float32)

    # Normalize
    density_hist = counts / total_pixels

    return density_hist


def preprocess_image(image):
    """
    Preprocesses a 2D CT slice for the EfficientNet backbone.
    1. Clips HU values to lung window [-1000, 400].
    2. Normalizes to [0, 1].
    3. Resizes to Config.IMG_SIZE.
    4. Stacks to 3 channels (RGB approximation).
    5. Transposes to (C, H, W) for PyTorch.

    Args:
        image (np.ndarray): 2D array of HU values.

    Returns:
        np.ndarray: Preprocessed image of shape (3, IMG_SIZE, IMG_SIZE).
    """
    target_size = Config.IMG_SIZE

    # Handle empty or zero inputs
    if image is None or image.size == 0:
        return np.zeros((3, target_size, target_size), dtype=np.float32)

    # 1. Clip to Lung Window
    # Standard lung window center is -600, width 1500 -> [-1350, 150]
    # However, for ML, a wider range covering soft tissue is often useful.
    # Using [-1000, 400] as a robust range covering air to soft tissue.
    img = np.clip(image, -1000, 400)

    # 2. Normalize to [0, 1]
    min_val = -1000.0
    max_val = 400.0
    img = (img - min_val) / (max_val - min_val)

    # 3. Resize
    # cv2.resize expects (width, height)
    try:
        img_resized = cv2.resize(
            img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )
    except Exception:
        # Fallback if resizing fails
        img_resized = np.zeros((target_size, target_size), dtype=np.float32)

    # 4. Stack to 3 channels
    # EfficientNet expects 3 channels. We duplicate the grayscale image.
    img_3c = np.stack([img_resized] * 3, axis=-1)  # (H, W, 3)

    # 5. Transpose to (C, H, W) for PyTorch
    img_transposed = np.transpose(img_3c, (2, 0, 1))

    return img_transposed.astype(np.float32)
