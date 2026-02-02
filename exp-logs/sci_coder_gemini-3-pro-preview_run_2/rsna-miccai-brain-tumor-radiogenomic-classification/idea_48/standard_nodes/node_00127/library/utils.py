import os
import numpy as np
import cv2
import random

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_seed()

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def read_dicom_robust(path):
    """
    Reads a DICOM file using OpenCV with a fallback to raw binary tail-reading.

    This function attempts to read the DICOM file using cv2.imread. If that fails
    (e.g., due to missing headers or format issues), it falls back to reading
    the raw binary data from the end of the file, assuming standard MRI dimensions
    and uint16 depth.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: Image array. Returns a zero array if all methods fail.
    """
    # Default fallback shape
    default_shape = (256, 256)

    if not os.path.exists(path):
        return np.zeros(default_shape, dtype=np.uint8)

    # 1. Primary Method: OpenCV
    try:
        # IMREAD_UNCHANGED preserves bit depth (e.g., uint16)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # 2. Fallback Method: Raw Binary Tail-Read
    # Assumes pixel data is located at the end of the file (common in DICOM).
    # We check against common square MRI dimensions.
    try:
        file_size = os.path.getsize(path)

        # Candidate dimensions (Height, Width) in descending order of size
        # We prioritize larger matches to avoid reading partial crops of larger images
        candidates = [
            (512, 512),
            (384, 384),
            (320, 320),
            (256, 256),
            (240, 240),
            (224, 224),
            (192, 192),
            (128, 128),
        ]

        for h, w in candidates:
            # Calculate expected pixel data size: H * W * 2 bytes (for uint16)
            pixel_bytes = h * w * 2

            # The file must be at least as large as the pixel data
            if file_size >= pixel_bytes:
                with open(path, "rb") as f:
                    # Seek to the start of the pixel data (end - pixel_bytes)
                    f.seek(-pixel_bytes, os.SEEK_END)
                    data = f.read(pixel_bytes)

                # Convert bytes to numpy array
                arr = np.frombuffer(data, dtype=np.uint16)

                # Verify size and reshape
                if arr.size == h * w:
                    return arr.reshape((h, w))

    except Exception:
        pass

    # 3. Final Fallback
    return np.zeros(default_shape, dtype=np.uint8)


def resize_image(img, size=(224, 224)):
    """
    Resizes an image using Area Interpolation.

    Area interpolation (cv2.INTER_AREA) is preferred for image shrinking as it
    resamples using pixel area relation, avoiding moire patterns and suppressing
    high-frequency noise better than Nearest Neighbor or Bilinear.

    Args:
        img (np.ndarray): Input image.
        size (tuple): Target size (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    if img is None or img.size == 0:
        # Return zero array of target size
        return np.zeros((size[1], size[0]), dtype=np.float32)

    try:
        # cv2.resize expects (width, height)
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return np.zeros((size[1], size[0]), dtype=np.float32)


def normalize_minmax(img):
    """
    Applies independent per-channel Min-Max scaling to the range [0, 1].

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized image (float32).
    """
    if img is None or img.size == 0:
        return np.array([], dtype=np.float32)

    # Ensure float32 for precision
    img = img.astype(np.float32)

    min_val = np.min(img)
    max_val = np.max(img)

    # Avoid division by zero
    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    else:
        # If image is constant (e.g., all zeros), return zeros
        return np.zeros_like(img, dtype=np.float32)
