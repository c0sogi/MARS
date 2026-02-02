import os
import glob
import re
import random
import numpy as np
import torch
import pydicom
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_windowing(image, center=Config.WINDOW_CENTER, width=Config.WINDOW_WIDTH):
    """
    Applies windowing to the image (converts to HU and normalizes).

    Args:
        image: Numpy array of the image (in HU).
        center: Window center.
        width: Window width.

    Returns:
        Windowed and normalized image [0, 1].
    """
    lower = center - width / 2
    upper = center + width / 2

    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)

    return image


def create_25d_stack(volume, index):
    """
    Creates a 2.5D stack of 3 slices (z-1, z, z+1) from the volume.

    Args:
        volume: 3D Numpy array (Depth, Height, Width).
        index: The index of the central slice.

    Returns:
        3D Numpy array of shape (3, Height, Width).
    """
    depth = volume.shape[0]

    # Determine indices for z-1, z, z+1 with boundary clamping
    idx_prev = max(0, index - 1)
    idx_curr = index
    idx_next = min(depth - 1, index + 1)

    # Stack the slices along the channel dimension (0)
    # volume is (D, H, W), we pick 3 slices to get (3, H, W)
    stack = np.stack([volume[idx_prev], volume[idx_curr], volume[idx_next]], axis=0)

    return stack


def load_dicom_stack(
    path,
    plane="axial",
    reverse_sort=False,
    load_cached_data=False,
    cache_dir=Config.CACHE_DIR,
):
    """
    Loads a stack of DICOM files into a 3D numpy array (Depth, Height, Width).

    Args:
        path: Path to the directory containing DICOM files OR a list of file paths.
        plane: Orientation (unused, assumes axial as per dataset).
        reverse_sort: Whether to reverse the sort order.
        load_cached_data: Whether to try loading from cache (only if path is a directory).
        cache_dir: Directory to store cached .npy files.

    Returns:
        Numpy array of shape (D, H, W) in Hounsfield Units (float32).
    """
    # Determine if we can use caching
    use_cache = False
    study_id = None
    cache_file = None

    if isinstance(path, str) and os.path.isdir(path):
        use_cache = load_cached_data
        # Assume directory name is the StudyInstanceUID
        study_id = os.path.basename(path.rstrip(os.sep))

        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{study_id}.npy")

            # Try to load from cache
            if os.path.exists(cache_file):
                try:
                    return np.load(cache_file)
                except Exception:
                    # If load fails, proceed to re-process
                    pass

    # Identify files
    files = []
    if isinstance(path, str) and os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.dcm"))
        # Sort numerically based on the filename (e.g., '10.dcm')
        # We extract the first sequence of digits found in the basename
        files.sort(key=lambda x: int(re.search(r"(\d+)", os.path.basename(x)).group(1)))
    elif isinstance(path, list):
        files = path
    else:
        # Return empty if path is invalid
        return np.zeros((0, 512, 512), dtype=np.float32)

    if reverse_sort:
        files = files[::-1]

    # Read DICOMs
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)

            # Get pixel array and convert to float
            img = ds.pixel_array.astype(np.float32)

            # Convert to Hounsfield Units (HU)
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                img = img * slope + intercept

            slices.append(img)
        except Exception:
            # Skip unreadable files
            continue

    if not slices:
        return np.zeros((0, 512, 512), dtype=np.float32)

    volume = np.stack(slices, axis=0)

    # Save to cache if enabled
    if use_cache and cache_file is not None:
        try:
            np.save(cache_file, volume)
        except Exception:
            pass

    return volume
