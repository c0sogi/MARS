import os
import re
import glob
import numpy as np
import cv2
from library.config import WORKING_DIR

# Attempt to import pydicom, fall back gracefully if not present
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def natural_sort_key(s):
    """
    Generates a key for natural sorting of filenames (e.g., Image-2.dcm before Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom_file(path):
    """
    Reads a DICOM file and returns the pixel array as a float32 numpy array.
    Prioritizes pydicom for accuracy, falls back to OpenCV.
    Returns None if reading fails.
    """
    # Method 1: pydicom (Standard for medical imaging)
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
            return img.astype(np.float32)
        except Exception:
            pass

    # Method 2: OpenCV (Fallback)
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    return None


def compute_roi_bounds(file_paths):
    """
    Scans the provided list of DICOM files to identify the start and end indices
    of the brain tissue (pixels > 0). This defines the Brain ROI.

    Args:
        file_paths (list): Sorted list of full file paths.

    Returns:
        tuple: (start_index, end_index)
    """
    if not file_paths:
        return 0, 0

    first_brain_idx = -1
    last_brain_idx = -1

    # Scan files to find the anatomical range
    for idx, path in enumerate(file_paths):
        img = read_dicom_file(path)
        if img is None:
            continue

        # Threshold check: any pixel > 0 implies tissue
        if np.max(img) > 0:
            if first_brain_idx == -1:
                first_brain_idx = idx
            last_brain_idx = idx

    # Fallback if no tissue found (e.g., all black images)
    if first_brain_idx == -1:
        return 0, len(file_paths) - 1

    return first_brain_idx, last_brain_idx


def get_relative_slice_index(roi_start, roi_end, relative_depth):
    """
    Calculates the integer slice index corresponding to a relative depth
    within the specific Brain ROI.

    Args:
        roi_start (int): Index where brain tissue starts.
        roi_end (int): Index where brain tissue ends.
        relative_depth (float): Target depth (0.0 to 1.0).

    Returns:
        int: The computed slice index clamped to bounds.
    """
    depth_range = roi_end - roi_start
    offset = int(depth_range * relative_depth)
    slice_idx = roi_start + offset

    # Ensure index is within valid bounds
    slice_idx = max(roi_start, min(slice_idx, roi_end))
    return slice_idx


def load_volumetric_stack(
    subject_id,
    subject_dir,
    modalities,
    relative_depths,
    img_size,
    load_cached_data=True,
):
    """
    Orchestrates the loading, processing, and stacking of the 9-channel input tensor.
    Implements caching to avoid re-computing ROI bounds.

    Args:
        subject_id (str/int): Unique identifier for the subject (used for cache naming).
        subject_dir (str): Full path to the subject's directory.
        modalities (list): List of modality names (e.g., ['FLAIR', 'T1wCE', 'T2w']).
        relative_depths (list): List of relative depths (e.g., [0.4, 0.5, 0.6]).
        img_size (int): Target spatial resolution (H, W).
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        np.ndarray: A tensor of shape (img_size, img_size, len(depths) * len(modalities)).
                    Order: Depth1[Mods], Depth2[Mods], Depth3[Mods].
    """
    # 1. Setup Cache Directory and Path
    cache_dir = os.path.join(WORKING_DIR, "cache_tensors")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{str(subject_id)}.npy")

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Cache corrupted or unreadable, proceed to recompute

    # 3. Process Data
    # Structure to hold images: modality -> depth -> image
    modality_slices = {m: {} for m in modalities}

    for mod in modalities:
        mod_dir = os.path.join(subject_dir, mod)

        # Handle missing directories
        if not os.path.exists(mod_dir):
            for depth in relative_depths:
                modality_slices[mod][depth] = np.zeros(
                    (img_size, img_size), dtype=np.float32
                )
            continue

        # Get sorted file list
        files = glob.glob(os.path.join(mod_dir, "*.dcm"))
        files.sort(key=natural_sort_key)

        if not files:
            for depth in relative_depths:
                modality_slices[mod][depth] = np.zeros(
                    (img_size, img_size), dtype=np.float32
                )
            continue

        # Compute ROI (Expensive operation)
        roi_start, roi_end = compute_roi_bounds(files)

        # Extract and Process Slices
        for depth in relative_depths:
            idx = get_relative_slice_index(roi_start, roi_end, depth)
            img_path = files[idx]

            img = read_dicom_file(img_path)

            if img is None:
                img = np.zeros((img_size, img_size), dtype=np.float32)
            else:
                # Resize
                if img.shape[0] != img_size or img.shape[1] != img_size:
                    img = cv2.resize(
                        img, (img_size, img_size), interpolation=cv2.INTER_AREA
                    )

                # Independent Channel Normalization [0, 1]
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)

            modality_slices[mod][depth] = img

    # 4. Stack Channels
    # Target Order: [Depth1_Mod1, Depth1_Mod2, Depth1_Mod3, Depth2_Mod1, ...]
    final_stack = []
    for depth in relative_depths:
        for mod in modalities:
            final_stack.append(modality_slices[mod][depth])

    # Stack along the last axis -> (H, W, 9)
    tensor = np.stack(final_stack, axis=-1)

    # 5. Save to Cache
    if load_cached_data:
        np.save(cache_path, tensor)

    return tensor
