import os
import glob
import numpy as np
import cv2
from library.config import Config

# Attempt to import pydicom to handle DICOM metadata (Slope/Intercept)
# If not available (not in the installed packages list), fallback to OpenCV with heuristics.
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def load_scan(study_uid, images_dir):
    """
    Loads a CT scan volume from DICOM files.

    Args:
        study_uid (str): The StudyInstanceUID.
        images_dir (str): The root directory containing study folders.

    Returns:
        np.ndarray: A 3D numpy array (Depth, Height, Width) containing pixel data.
                    Data is converted to Hounsfield Units (HU) if possible,
                    or raw values are adjusted heuristically.
    """
    scan_path = os.path.join(images_dir, study_uid)
    if not os.path.isdir(scan_path):
        raise FileNotFoundError(f"Study directory not found: {scan_path}")

    # List all DICOM files
    files = glob.glob(os.path.join(scan_path, "*.dcm"))
    if not files:
        # Fallback for files without extension
        files = glob.glob(os.path.join(scan_path, "*"))

    if not files:
        raise FileNotFoundError(f"No files found in {scan_path}")

    # Sort files based on Instance Number (filename).
    # This assumes filenames are like '1.dcm', '10.dcm' which correspond to Z-position.
    def get_file_index(filepath):
        name = os.path.basename(filepath).split(".")[0]
        return int(name) if name.isdigit() else 0

    files.sort(key=get_file_index)

    slices = []

    if HAS_PYDICOM:
        # Use pydicom for accurate HU conversion
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                slope = float(getattr(ds, "RescaleSlope", 1))
                intercept = float(getattr(ds, "RescaleIntercept", -1024))

                img = ds.pixel_array.astype(np.float32)
                img = img * slope + intercept
                slices.append(img)
            except Exception:
                continue
    else:
        # Fallback to OpenCV
        # This path is taken if pydicom is not installed in the environment.
        for f in files:
            # IMREAD_UNCHANGED preserves bit-depth (e.g., 12/16-bit CT data)
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            img = img.astype(np.float32)

            # Heuristic: CT data is often stored as unsigned integers with an offset.
            # Standard CT air is -1000 HU. If data is shifted by +1024 (common), min value is ~0.
            # We subtract 1024 to approximate HU.
            if np.min(img) >= 0:
                img -= 1024.0

            slices.append(img)

    if not slices:
        raise ValueError(f"Failed to load any slices for study {study_uid}")

    # Stack to create volume (Depth, H, W)
    volume = np.stack(slices)
    return volume


def window_image(volume, window_center, window_width):
    """
    Applies standard CT windowing and normalizes to 0-255 uint8.

    Args:
        volume (np.ndarray): Input volume in HU.
        window_center (float): Window Level (e.g., 400 for bone).
        window_width (float): Window Width (e.g., 1800 for bone).

    Returns:
        np.ndarray: Windowed volume cast to uint8.
    """
    min_val = window_center - window_width / 2.0
    max_val = window_center + window_width / 2.0

    # Clip to window
    volume = np.clip(volume, min_val, max_val)

    # Normalize to [0, 1]
    volume = (volume - min_val) / (max_val - min_val)

    # Scale to [0, 255] and cast
    volume = (volume * 255.0).astype(np.uint8)

    return volume


def resize_volume(volume, size):
    """
    Resizes the volume slice-by-slice to the target spatial dimensions.

    Args:
        volume (np.ndarray): Input volume (Depth, H, W).
        size (int): Target height/width.

    Returns:
        np.ndarray: Resized volume (Depth, size, size).
    """
    # Avoid resizing if already correct
    if volume.shape[1] == size and volume.shape[2] == size:
        return volume

    resized_slices = []
    for i in range(volume.shape[0]):
        # cv2.resize expects (width, height)
        sl = cv2.resize(volume[i], (size, size), interpolation=cv2.INTER_LINEAR)
        resized_slices.append(sl)

    return np.stack(resized_slices)


def process_scan(study_uid, images_dir, load_cached_data=True):
    """
    Orchestrates the loading, windowing, resizing, and caching of a scan.
    Implements the deterministic data processing pipeline.

    Args:
        study_uid (str): The unique study identifier.
        images_dir (str): Directory containing the study images.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        np.ndarray: The processed volume (Depth, H, W) in uint8 format.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_uid}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            # If cache is corrupt or unreadable, fall through to recompute
            pass

    # 2. Process from scratch
    # Load raw data (HU)
    volume = load_scan(study_uid, images_dir)

    # Apply Bone Window
    volume = window_image(volume, Config.WINDOW_LEVEL, Config.WINDOW_WIDTH)

    # Resize to model input size (saves disk space and memory)
    volume = resize_volume(volume, Config.IMAGE_SIZE)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, volume)

    return volume


def get_25d_stack(volume, slice_idx):
    """
    Extracts a 2.5D stack (3 channels) for a specific slice index.
    The stack consists of [slice_z-1, slice_z, slice_z+1].

    Args:
        volume (np.ndarray): The processed 3D volume (Depth, H, W).
        slice_idx (int): The index of the center slice.

    Returns:
        np.ndarray: A 3-channel image of shape (H, W, 3).
    """
    depth, h, w = volume.shape

    # Handle boundary conditions by clamping indices
    idx_prev = max(0, slice_idx - 1)
    idx_curr = slice_idx
    idx_next = min(depth - 1, slice_idx + 1)

    # Extract slices
    s_prev = volume[idx_prev]
    s_curr = volume[idx_curr]
    s_next = volume[idx_next]

    # Stack along the last dimension (H, W, C) for compatibility with albumentations/cv2
    stack = np.stack([s_prev, s_curr, s_next], axis=-1)

    return stack
