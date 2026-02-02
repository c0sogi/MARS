import os
import numpy as np
import cv2
from library.config import Config


def load_scan(dcm_dir):
    """
    Loads CT scan from a directory of DICOM files.
    Since pydicom is not available in the allowed packages list,
    this function uses OpenCV and filename sorting.

    Applies a heuristic for HU conversion: HU = PixelValue - 1024.
    """
    if not os.path.exists(dcm_dir):
        return None

    # List .dcm files
    files = [f for f in os.listdir(dcm_dir) if f.lower().endswith(".dcm")]
    if not files:
        return None

    # Sort by instance number derived from filename (e.g., '10.dcm' -> 10)
    # This ensures correct Z-ordering
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()  # Fallback to lexicographical if filenames are not integers

    slices = []
    for f in files:
        path = os.path.join(dcm_dir, f)
        # Read image using OpenCV with IMREAD_UNCHANGED to preserve depth (usually 16-bit)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # Handle cases where image might be loaded as 3-channel
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        slices.append(img)

    if not slices:
        return None

    # Stack into 3D volume (Z, H, W)
    volume = np.stack(slices, axis=0).astype(np.float32)

    # Heuristic HU conversion
    # Standard CT scanners have an intercept of -1024
    # This shifts the unsigned integer data to the Hounsfield scale
    volume = volume - 1024.0

    return volume


def segment_lung(volume):
    """
    Segments lung tissue using HU thresholds.
    Sets non-lung pixels to HU_MIN (-1000) to mask them out.
    """
    # Define thresholds
    hu_min = Config.HU_MIN
    hu_max = Config.HU_MAX

    # Create binary mask for lung tissue
    mask = (volume >= hu_min) & (volume <= hu_max)

    # Create a copy to return
    masked_volume = volume.copy()

    # Set background (air) and high-density structures (bone) to HU_MIN
    # Setting to HU_MIN ensures they don't interfere with MIP (Max Intensity Projection)
    # or Variance calculation (providing a constant background)
    masked_volume[~mask] = hu_min

    # Clip values to range to ensure clean normalization later
    masked_volume = np.clip(masked_volume, hu_min, hu_max)

    return masked_volume


def resize_and_normalize(image):
    """
    Resizes image to Config.IMAGE_SIZE and normalizes to [0, 1].
    """
    # Resize using linear interpolation
    img_resized = cv2.resize(
        image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )

    # Normalize to [0, 1] based on fixed HU range
    min_val = Config.HU_MIN
    max_val = Config.HU_MAX

    # Avoid division by zero
    denom = max_val - min_val if max_val != min_val else 1.0

    img_norm = (img_resized - min_val) / denom
    img_norm = np.clip(img_norm, 0.0, 1.0)

    return img_norm.astype(np.float32)


def compute_coronal_mip(volume):
    """
    Computes Coronal Maximum Intensity Projection (MIP).
    Input Volume: (Z, H, W)
    Coronal Projection: Project along Y-axis (H) -> Result (Z, W)
    """
    # Compute Max over axis 1 (Height/Y-axis)
    mip = np.max(volume, axis=1)

    return resize_and_normalize(mip)


def select_zonal_slices(volume):
    """
    Selects 3 axial slices (Upper, Middle, Lower) based on maximum variance.
    Input Volume: (Z, H, W)
    """
    z_depth = volume.shape[0]

    # Define split points for 3 zones
    splits = [0, z_depth // 3, (z_depth * 2) // 3, z_depth]

    selected_slices = []

    for i in range(3):
        start, end = splits[i], splits[i + 1]

        # Handle cases with very few slices
        if start >= end:
            # Fallback: take the last available slice or middle
            idx = min(start, z_depth - 1)
            slice_img = volume[idx]
        else:
            zone_chunk = volume[start:end]

            # Calculate variance for each slice in the chunk
            # axis=(1, 2) computes variance of all pixels in the 2D slice
            variances = np.var(zone_chunk, axis=(1, 2))

            # Pick index of max variance (indicates most tissue/structure)
            max_var_idx = np.argmax(variances)
            slice_img = zone_chunk[max_var_idx]

        selected_slices.append(resize_and_normalize(slice_img))

    return selected_slices


def compute_density_histogram(volume):
    """
    Computes density histogram of the lung volume.
    """
    # Flatten volume
    flat_vol = volume.flatten()

    # Compute histogram
    # Range is strictly HU_MIN to HU_MAX
    hist, _ = np.histogram(
        flat_vol,
        bins=Config.HISTOGRAM_BINS,
        range=(Config.HU_MIN, Config.HU_MAX),
        density=True,
    )

    return hist.astype(np.float32)


def process_patient(patient_id, dcm_path, load_cached_data=True):
    """
    Orchestrates the processing pipeline for a single patient.

    Args:
        patient_id (str): Unique patient identifier.
        dcm_path (str): Relative path to DICOM directory (e.g., 'train/ID...').
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: Dictionary containing processed arrays ('mip', 'axial_1', 'axial_2', 'axial_3', 'histogram').
    """
    # Define cache path
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return {
                "mip": data["mip"],
                "axial_1": data["axial_1"],
                "axial_2": data["axial_2"],
                "axial_3": data["axial_3"],
                "histogram": data["histogram"],
            }
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from Scratch
    full_path = os.path.join(Config.INPUT_DIR, dcm_path)

    volume = load_scan(full_path)

    # Handle missing or corrupt data
    if volume is None:
        zeros_img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
        zeros_hist = np.zeros(Config.HISTOGRAM_BINS, dtype=np.float32)
        return {
            "mip": zeros_img,
            "axial_1": zeros_img,
            "axial_2": zeros_img,
            "axial_3": zeros_img,
            "histogram": zeros_hist,
        }

    # Process Volume
    masked_vol = segment_lung(volume)

    # Generate Features
    mip = compute_coronal_mip(masked_vol)
    axials = select_zonal_slices(masked_vol)
    hist = compute_density_histogram(masked_vol)

    results = {
        "mip": mip,
        "axial_1": axials[0],
        "axial_2": axials[1],
        "axial_3": axials[2],
        "histogram": hist,
    }

    # 3. Save Cache
    # Ensure directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Use savez_compressed for efficiency
    np.savez_compressed(cache_path, **results)

    return results
