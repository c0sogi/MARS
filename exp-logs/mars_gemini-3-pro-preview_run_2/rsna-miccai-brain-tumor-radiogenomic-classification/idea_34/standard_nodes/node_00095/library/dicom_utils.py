import os
import re
import numpy as np
import cv2


def get_slice_id(filename):
    """
    Extracts the integer slice ID from a DICOM filename.
    Expected format: 'Image-{id}.dcm'
    """
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    return -1


def map_slice_ids(directory):
    """
    Scans a directory for DICOM files and creates a mapping from slice ID to file path.
    This enables 'Explicit Alignment' by ID rather than sorting by filename.
    """
    mapping = {}
    if not os.path.exists(directory):
        return mapping

    for f in os.listdir(directory):
        if f.lower().endswith(".dcm"):
            sid = get_slice_id(f)
            if sid != -1:
                mapping[sid] = os.path.join(directory, f)
    return mapping


def read_dicom_robust(filepath, target_size=None):
    """
    Reads a DICOM file with a robust fallback strategy.

    1. Attempts to read using OpenCV (standard method).
    2. If that fails (e.g., corrupt header), falls back to 'Raw Binary Tail-Read'.
       - Assumes 16-bit depth.
       - Infers resolution (512x512 vs 256x256) based on file size.
       - Reads pixel data from the end of the file.

    Args:
        filepath (str): Path to the DICOM file.
        target_size (tuple, optional): (width, height) to resize the output to.
                                       Uses cv2.INTER_AREA for downsampling.

    Returns:
        np.ndarray: 2D image array of type float32.
    """
    img = None

    # Strategy 1: Standard OpenCV Read
    if os.path.exists(filepath):
        try:
            # -1 loads as-is (16-bit if available)
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None

    # Strategy 2: Raw Binary Tail-Read (Fallback)
    if img is None and os.path.exists(filepath):
        try:
            file_size = os.path.getsize(filepath)

            # Heuristic: BraTS data is typically 512x512 or 256x256
            # 512*512*2 bytes = 524,288 bytes (~512KB)
            # 256*256*2 bytes = 131,072 bytes (~128KB)
            # We use a threshold of 200KB to distinguish.

            with open(filepath, "rb") as f:
                raw_data = f.read()

            if file_size > 200 * 1024:
                # Assume 512x512
                num_pixels = 512 * 512
                if len(raw_data) >= num_pixels * 2:
                    # Read last N bytes
                    pixel_data = raw_data[-num_pixels * 2 :]
                    img = np.frombuffer(pixel_data, dtype="<u2").reshape((512, 512))
            else:
                # Assume 256x256
                num_pixels = 256 * 256
                if len(raw_data) >= num_pixels * 2:
                    pixel_data = raw_data[-num_pixels * 2 :]
                    img = np.frombuffer(pixel_data, dtype="<u2").reshape((256, 256))

        except Exception:
            img = None

    # Handle Failure: Return zeros if everything failed
    if img is None:
        # If target size is known, return zeros of that size, else default 224
        h, w = target_size if target_size else (224, 224)
        return np.zeros((h, w), dtype=np.float32)

    # Convert to float32 (Precision requirement)
    img = img.astype(np.float32)

    # Resize if requested
    if target_size is not None:
        current_h, current_w = img.shape[:2]
        target_w, target_h = target_size

        if (current_w, current_h) != (target_w, target_h):
            # Use INTER_AREA for shrinking (denoising), LINEAR for enlarging
            interpolation = (
                cv2.INTER_AREA if (current_w > target_w) else cv2.INTER_LINEAR
            )
            img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    return img
