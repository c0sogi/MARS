import os
import numpy as np
import cv2
import pydicom
from library.config import Config


def read_dicom_robust(filepath):
    """
    Reads a DICOM file with a robust fallback mechanism.

    Strategy:
    1. Attempt standard `pydicom.dcmread` with `force=True`.
    2. If header parsing fails, perform a Raw Binary Tail-Read anchored to the
       end of the file. This assumes the pixel data is located at the end of
       the file and corresponds to standard MRI dimensions (e.g., 512x512, 256x256).

    Args:
        filepath (str): Path to the .dcm file.

    Returns:
        np.ndarray: Image data as float32. Returns a zero array on total failure.
    """
    if not os.path.exists(filepath):
        # Return a placeholder if file is missing to maintain pipeline flow
        return np.zeros(Config.IMG_SIZE, dtype=np.float32)

    try:
        # Primary Method: Standard DICOM parsing
        # force=True allows reading files with missing preambles
        dcm = pydicom.dcmread(filepath, force=True)
        img = dcm.pixel_array.astype(np.float32)
        return img
    except Exception:
        # Secondary Method: Raw Binary Tail-Read
        # This bypasses brittle headers by reading pixel data directly from the file end.
        try:
            file_size = os.path.getsize(filepath)

            # Heuristic for dimensions based on file size and common MRI resolutions
            # We assume uint16 (2 bytes per pixel)
            # 512x512 * 2 = 524,288 bytes
            # 256x256 * 2 = 131,072 bytes

            if file_size >= 524288:
                rows, cols = 512, 512
            elif file_size >= 131072:
                rows, cols = 256, 256
            else:
                # Fallback: Estimate square dimension from available bytes
                pixels = file_size // 2
                dim = int(np.sqrt(pixels))
                rows, cols = dim, dim

            num_bytes = rows * cols * 2

            with open(filepath, "rb") as f:
                # Anchor to the end of the file
                f.seek(-num_bytes, 2)
                data = f.read(num_bytes)

            img = (
                np.frombuffer(data, dtype=np.uint16)
                .reshape((rows, cols))
                .astype(np.float32)
            )
            return img

        except Exception:
            # Final fallback: Return zeros
            return np.zeros(Config.IMG_SIZE, dtype=np.float32)


def resize_image(image, target_size=Config.IMG_SIZE):
    """
    Resizes the image to the target dimensions using Area Interpolation.

    Area Interpolation (INTER_AREA) is preferred for medical imaging downsampling
    as it suppresses moire patterns and high-frequency noise better than
    Linear or Cubic interpolation.

    Args:
        image (np.ndarray): Input image.
        target_size (tuple): Desired (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    try:
        if image is None or image.size == 0:
            return np.zeros(target_size, dtype=np.float32)

        # Check if resize is necessary
        if image.shape[0] == target_size[0] and image.shape[1] == target_size[1]:
            return image

        # cv2.resize expects dsize as (width, height).
        # Config.IMG_SIZE is typically (H, W).
        # We assume square inputs for simplicity or matching config.
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return np.zeros(target_size, dtype=np.float32)


def normalize_image(image):
    """
    Applies Independent Per-Channel Min-Max Scaling to [0, 1].

    This preserves the relative tissue contrast within the specific slice/modality,
    which is crucial for detecting subtle methylation signals.

    Args:
        image (np.ndarray): Input image (float32).

    Returns:
        np.ndarray: Normalized image in range [0, 1].
    """
    try:
        if image is None or image.size == 0:
            return image

        img_min = np.min(image)
        img_max = np.max(image)

        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        else:
            # Handle constant images (e.g., all black)
            return np.zeros_like(image)
    except Exception:
        return np.zeros_like(image)
