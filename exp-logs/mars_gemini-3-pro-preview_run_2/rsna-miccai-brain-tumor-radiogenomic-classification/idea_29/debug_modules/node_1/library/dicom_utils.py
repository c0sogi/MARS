import os
import numpy as np
import cv2
from library.config import IMG_SIZE


def read_dicom_robust(path):
    """
    Reads a DICOM file using a raw binary tail-read strategy to bypass header parsing.
    This is necessary because pydicom is not available and headers can be brittle.

    Assumes uncompressed pixel data located at the end of the file.
    Supports 512x512 and 256x256 resolutions based on file size heuristics.

    Args:
        path (str): Path to the DICOM file.

    Returns:
        np.ndarray: 2D numpy array of the image (uint16), or a blank array on failure.
    """
    try:
        if not os.path.exists(path):
            return np.zeros((256, 256), dtype=np.uint16)

        file_size = os.path.getsize(path)

        # Define expected pixel data sizes (16-bit depth = 2 bytes per pixel)
        size_512 = 512 * 512 * 2  # 524,288 bytes
        size_256 = 256 * 256 * 2  # 131,072 bytes

        # Heuristic to determine resolution and read offset
        # Files are typically ~525KB (512x512) or ~132KB (256x256) with small headers
        if file_size >= size_512:
            rows, cols = 512, 512
            num_bytes = size_512
        elif file_size >= size_256:
            rows, cols = 256, 256
            num_bytes = size_256
        else:
            # File too small to contain a standard MRI slice
            return np.zeros((256, 256), dtype=np.uint16)

        with open(path, "rb") as f:
            # Seek to the end minus the pixel data size to skip the header
            f.seek(-num_bytes, 2)
            buffer = f.read(num_bytes)

        if len(buffer) != num_bytes:
            return np.zeros((rows, cols), dtype=np.uint16)

        # Convert buffer to numpy array
        img = np.frombuffer(buffer, dtype=np.uint16)

        # Reshape to image dimensions
        img = img.reshape((rows, cols))

        return img

    except Exception as e:
        # Return blank image on any IO or parsing error to maintain pipeline stability
        return np.zeros((256, 256), dtype=np.uint16)


def process_image(img):
    """
    Processes a raw image array for model ingestion.
    Converts to float32 and resizes to the target IMG_SIZE using Area Interpolation.

    Args:
        img (np.ndarray): Input 2D image array (uint16).

    Returns:
        np.ndarray: Processed 2D image array (float32) of shape (IMG_SIZE, IMG_SIZE).
    """
    # Convert to float32
    img = img.astype(np.float32)

    # Resize to target dimensions
    # cv2.INTER_AREA is recommended for image decimation (downsampling)
    # as it suppresses moire patterns and high-frequency noise.
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return img
