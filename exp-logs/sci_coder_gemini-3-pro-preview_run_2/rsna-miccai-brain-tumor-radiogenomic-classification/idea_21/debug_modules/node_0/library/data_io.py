import os
import numpy as np
import cv2
from library.config import Config


def read_dicom_robust(file_path):
    """
    Reads a DICOM file using OpenCV, falling back to a raw binary tail-read
    if the header is corrupt or unreadable.

    Args:
        file_path (str): Path to the .dcm file.

    Returns:
        np.ndarray: The image array (uint16), or a zero array on total failure.
    """
    if not os.path.exists(file_path):
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)

    # 1. Try Standard Loading via OpenCV
    try:
        # IMREAD_UNCHANGED is critical to preserve 16-bit depth of MRI scans
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Handle rare cases where single-channel DICOM is read as 3-channel
            if img.ndim == 3:
                img = img[:, :, 0]
            return img
    except Exception:
        pass

    # 2. Fallback: Raw Binary Tail-Read
    # This bypasses corrupt headers by reading raw pixel bytes from the end of the file.
    # We infer resolution (512x512 or 256x256) based on file size heuristics.
    try:
        file_size = os.path.getsize(file_path)

        # Heuristics:
        # 512 * 512 * 2 bytes (uint16) = 524,288 bytes
        # 256 * 256 * 2 bytes (uint16) = 131,072 bytes
        # We check if file is large enough to contain these raw buffers.

        if file_size >= 524288:
            rows, cols = 512, 512
            num_bytes = 524288
        elif file_size >= 131072:
            rows, cols = 256, 256
            num_bytes = 131072
        else:
            # File too small to be a valid scan slice
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)

        with open(file_path, "rb") as f:
            # Seek to the start of the pixel data (End - RawBytes)
            f.seek(-num_bytes, 2)
            buffer = f.read(num_bytes)

        img_array = np.frombuffer(buffer, dtype=np.uint16)

        if img_array.size == rows * cols:
            return img_array.reshape((rows, cols))

    except Exception:
        pass

    # Return black image on total failure to prevent pipeline crash
    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)


def resize_image(image, target_size=Config.IMG_SIZE):
    """
    Resizes an image to the target size using Area Interpolation to prevent aliasing.
    Casts output to float32 for model consumption.

    Args:
        image (np.ndarray): Input image (any resolution).
        target_size (int): Target width/height.

    Returns:
        np.ndarray: Resized image as float32.
    """
    if image is None or image.size == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Ensure input is 2D
    if image.ndim > 2:
        image = image[:, :, 0]

    # Optimization: skip resize if already correct
    if image.shape[0] == target_size and image.shape[1] == target_size:
        return image.astype(np.float32)

    # Use INTER_AREA for downsampling (e.g., 512 -> 224) to suppress Moiré patterns/noise
    resized = cv2.resize(
        image, (target_size, target_size), interpolation=cv2.INTER_AREA
    )

    return resized.astype(np.float32)


def load_and_preprocess_slice(file_path):
    """
    High-level wrapper to load a DICOM file robustly and resize it.

    Args:
        file_path (str): Path to the DICOM file.

    Returns:
        np.ndarray: Preprocessed image (float32, target_size x target_size).
    """
    raw_img = read_dicom_robust(file_path)
    processed_img = resize_image(raw_img, target_size=Config.IMG_SIZE)
    return processed_img
