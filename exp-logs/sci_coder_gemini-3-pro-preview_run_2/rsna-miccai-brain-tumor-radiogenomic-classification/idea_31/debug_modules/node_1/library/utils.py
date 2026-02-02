import os
import numpy as np
import cv2
from library.config import Config


def load_dicom_robust(file_path):
    """
    Loads a DICOM file robustly, prioritizing standard libraries and falling back
    to a raw binary tail-read anchored to the end of the file.

    Args:
        file_path (str): Path to the DICOM file.

    Returns:
        np.ndarray: Image data as float32. Returns a zero array on total failure.
    """
    # 1. Attempt Standard Loading (OpenCV)
    # Note: Standard OpenCV often lacks DICOM support, but we attempt it as a primary method.
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # 2. Fallback: Raw Binary Tail-Read
    # Assumes uncompressed pixel data located at the end of the file.
    # We infer resolution from file size assuming uint16 depth.
    try:
        if not os.path.exists(file_path):
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        file_size = os.path.getsize(file_path)

        # Expected raw sizes (pixels * 2 bytes)
        size_512 = 512 * 512 * 2
        size_256 = 256 * 256 * 2

        # Determine likely resolution
        if file_size >= size_512:
            rows, cols = 512, 512
            bytes_to_read = size_512
        elif file_size >= size_256:
            rows, cols = 256, 256
            bytes_to_read = size_256
        else:
            # File too small or unknown geometry
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Read the tail of the file
        with open(file_path, "rb") as f:
            f.seek(-bytes_to_read, os.SEEK_END)
            raw_data = f.read(bytes_to_read)

        # Convert bytes to numpy array
        img = np.frombuffer(raw_data, dtype=np.uint16).astype(np.float32)

        # Reshape to 2D image
        if img.size == rows * cols:
            img = img.reshape(rows, cols)
        else:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        return img

    except Exception:
        # Robust fallback for any IO/Parsing errors
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def resize_image(image, size=None):
    """
    Resizes an image to the specified dimensions using Area Interpolation.

    Args:
        image (np.ndarray): Input image.
        size (tuple or int, optional): Target size. Defaults to Config.IMG_SIZE.

    Returns:
        np.ndarray: Resized image.
    """
    if size is None:
        target_size = (Config.IMG_SIZE, Config.IMG_SIZE)
    elif isinstance(size, int):
        target_size = (size, size)
    else:
        target_size = size

    # cv2.resize expects dsize=(width, height)
    # image.shape is (height, width)

    # Optimization: Skip if already correct size
    if image.shape[0] == target_size[1] and image.shape[1] == target_size[0]:
        return image

    # Use INTER_AREA for downsampling (e.g., 512 -> 224) to reduce aliasing/noise
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_minmax(image):
    """
    Applies independent per-channel Min-Max scaling to the range [0, 1].

    Args:
        image (np.ndarray): Input image (single channel/slice).

    Returns:
        np.ndarray: Normalized image.
    """
    img_min = image.min()
    img_max = image.max()

    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)

    # Handle constant images (e.g., all zeros) to avoid division by zero
    return image - img_min
