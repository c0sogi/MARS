import os
import numpy as np
import cv2
import torch
from library.config import Config

# Attempt to import pydicom. If not available, we will fallback to dummy data.
# This ensures the code runs even if the specific DICOM library is missing in the environment.
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom not found. Image processing will yield zero-arrays.")


def get_pixels_hu(ds):
    """
    Converts a pydicom dataset's pixel_array to Hounsfield Units (HU).
    Handles RescaleSlope and RescaleIntercept.
    """
    try:
        image = ds.pixel_array.astype(np.float32)

        # Intercept and Slope
        intercept = getattr(ds, "RescaleIntercept", -1024)
        slope = getattr(ds, "RescaleSlope", 1)

        # Handle cases where metadata might be lists/arrays
        if isinstance(slope, (list, np.ndarray)):
            slope = slope[0]
        if isinstance(intercept, (list, np.ndarray)):
            intercept = intercept[0]

        image = image * slope + intercept
        return image
    except Exception:
        # Fallback for corrupt data
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def calculate_lung_area(ds):
    """
    Calculates the approximate lung area in a slice using a threshold.
    Lungs are air-filled (< -320 HU).
    """
    try:
        hu_image = get_pixels_hu(ds)
        # Threshold: Air is ~ -1000 HU. Lung tissue is < -300 HU.
        # We use < -320 as a rough segmentation mask for lung+air.
        binary = (hu_image < -320).astype(np.uint8)
        return np.sum(binary)
    except Exception:
        return 0.0


def load_scans(scan_dir):
    """
    Loads all DICOM files from a directory and sorts them by Z-position.
    """
    if not HAS_PYDICOM or not os.path.exists(scan_dir):
        return []

    slices = []
    # List files
    try:
        files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".dcm")]
    except OSError:
        return []

    for s in files:
        try:
            full_path = os.path.join(scan_dir, s)
            # Use stop_before_pixels=False because we need pixels for area calc
            ds = pydicom.dcmread(full_path)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return []

    # Sort by ImagePositionPatient[2] (Z-axis)
    # If missing, fallback to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass  # Keep file system order (unreliable but better than crash)

    return slices


def select_adaptive_slices(scans):
    """
    Selects 3 slices based on lung area:
    1. Anchor: Slice with maximum lung area.
    2. Top: First slice above Anchor with area < 50% of max.
    3. Bottom: First slice below Anchor with area < 50% of max.
    """
    num_slices = len(scans)
    if num_slices == 0:
        return [None] * 3

    if num_slices < 3:
        # Not enough slices, replicate the first one
        return [scans[0]] * 3

    # Compute areas
    areas = [calculate_lung_area(s) for s in scans]
    max_area = np.max(areas)

    if max_area == 0:
        # No lung detected (or empty scans), return middle slices
        mid = num_slices // 2
        return [scans[max(0, mid - 1)], scans[mid], scans[min(num_slices - 1, mid + 1)]]

    anchor_idx = np.argmax(areas)
    threshold = 0.5 * max_area

    # Find Top (indices < anchor_idx)
    # We search backwards from anchor to finding the start of the lungs
    top_idx = 0
    for i in range(anchor_idx, -1, -1):
        if areas[i] < threshold:
            top_idx = i
            break

    # Find Bottom (indices > anchor_idx)
    # We search forwards from anchor to finding the end of the lungs
    bottom_idx = num_slices - 1
    for i in range(anchor_idx, num_slices):
        if areas[i] < threshold:
            bottom_idx = i
            break

    return [scans[top_idx], scans[anchor_idx], scans[bottom_idx]]


def preprocess_slice(ds):
    """
    Applies windowing, normalization, and resizing to a DICOM slice.
    """
    if ds is None:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    try:
        # 1. Convert to HU
        image = get_pixels_hu(ds)

        # 2. Windowing
        # Lung Window: typically W=1500, L=-600 -> [-1350, 150]
        # We use a broad range [-1000, 400] to capture lung and some soft tissue
        min_bound = -1000.0
        max_bound = 400.0
        image = np.clip(image, min_bound, max_bound)

        # 3. Normalize to [0, 1]
        image = (image - min_bound) / (max_bound - min_bound)

        # 4. Resize
        # cv2.resize expects (W, H) but since it's square it doesn't matter
        image = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))

        return image.astype(np.float32)

    except Exception:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def process_patient(patient_id, relative_path, load_cached_data=True):
    """
    Main pipeline function.

    Args:
        patient_id (str): Unique patient identifier.
        relative_path (str): Path to DICOM directory relative to INPUT_DIR.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.array: Shape (3, IMG_SIZE, IMG_SIZE), float32.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            if data.shape == (3, Config.IMG_SIZE, Config.IMG_SIZE):
                return data
        except Exception:
            pass  # Corrupt cache, recompute

    # 2. Compute from scratch
    full_path = os.path.join(Config.INPUT_DIR, relative_path)

    # Load and Select
    scans = load_scans(full_path)
    selected_scans = select_adaptive_slices(scans)

    # Preprocess
    processed_slices = []
    for s in selected_scans:
        processed_slices.append(preprocess_slice(s))

    # Stack -> (3, H, W)
    data = np.stack(processed_slices, axis=0)

    # 3. Save Cache
    try:
        np.save(cache_file, data)
    except Exception:
        pass  # Disk write error

    return data
