import os
import numpy as np
import cv2
from library.config import Config


def read_dicom_robust(file_path):
    """
    Reads a DICOM file using a raw binary tail-read method to bypass
    brittle header parsing issues. This method assumes the pixel data
    is stored as uint16 at the end of the file and corresponds to a
    square image of standard dimensions.

    This function is designed to work without pydicom, relying on
    file size heuristics to determine the image resolution.

    Args:
        file_path (str): Path to the DICOM file.

    Returns:
        np.ndarray: The image data as a numpy array (uint16).
                    Returns a zero-filled array if reading fails.
    """
    # Default fallback image (black)
    default_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)

    if not os.path.exists(file_path):
        return default_img

    try:
        file_size = os.path.getsize(file_path)

        # Standard DICOM image dimensions to check (Square)
        # Ordered from largest to smallest to find the best fit.
        # 1024x1024x2 = 2MB, 512x512x2 = 512KB, 256x256x2 = 128KB
        candidate_sizes = [1024, 512, 384, 320, 256, 224, 192, 128, 64]

        best_size = 0
        min_diff = float("inf")

        # Heuristic: The file size must be at least the size of the pixel data.
        # The 'diff' (header size) should be minimal but positive.
        for size in candidate_sizes:
            needed_bytes = size * size * 2
            diff = file_size - needed_bytes

            # We assume the header is not excessively large (e.g., < 50KB)
            # and that the pixel data is the dominant part of the file.
            if 0 <= diff < min_diff:
                min_diff = diff
                best_size = size

        # If no suitable size is found, return default
        if best_size == 0:
            return default_img

        needed_bytes = best_size * best_size * 2

        with open(file_path, "rb") as f:
            # Seek to the start of the pixel data relative to the end of the file
            f.seek(-needed_bytes, 2)
            buffer = f.read(needed_bytes)

        # Convert buffer to numpy array
        # DICOM pixel data is typically Little Endian 16-bit unsigned integers
        img = np.frombuffer(buffer, dtype="<u2").copy()
        img = img.reshape((best_size, best_size))

        return img

    except Exception:
        # Return default zero image on any error to ensure pipeline robustness
        return default_img


def preprocess_image(image):
    """
    Preprocesses the raw image data:
    1. Converts to float32.
    2. Resizes to the target configuration size using Area Interpolation.

    Args:
        image (np.ndarray): Input image (uint16 or other).

    Returns:
        np.ndarray: Preprocessed image (float32, resized).
    """
    # Convert to float32 to preserve precision during subsequent operations
    img_float = image.astype(np.float32)

    # Resize if dimensions do not match target
    if img_float.shape[0] != Config.IMG_SIZE or img_float.shape[1] != Config.IMG_SIZE:
        # cv2.INTER_AREA is optimal for downsampling (decimation) as it resamples
        # using pixel area relation, avoiding moiré patterns and high-frequency noise.
        img_resized = cv2.resize(
            img_float, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        return img_resized

    return img_float
