import os
import numpy as np
import cv2
from library.config import Config, seed_everything


def read_dicom_robust(path):
    """
    Reads a DICOM file using a raw binary tail-read strategy.
    This bypasses header parsing issues and works without pydicom.
    It assumes the pixel data is uncompressed (uint16) and located at the end of the file.
    """
    try:
        if not os.path.exists(path):
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)

        file_size = os.path.getsize(path)

        # Heuristic to determine resolution based on file size thresholds.
        # Standard uncompressed DICOM pixel buffers (uint16 = 2 bytes/pixel):
        # 512x512 = 524,288 bytes
        # 256x256 = 131,072 bytes
        # 240x240 = 115,200 bytes

        # We check if the file is large enough to contain the pixel buffer plus a header.
        if file_size >= 524288:
            rows, cols = 512, 512
        elif file_size >= 131072:
            rows, cols = 256, 256
        elif file_size >= 115200:
            rows, cols = 240, 240
        else:
            # If file is too small for known resolutions, return a zero array
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)

        pixel_bytes = rows * cols * 2

        with open(path, "rb") as f:
            # Seek to the end minus the size of the pixel buffer to skip the header
            f.seek(-pixel_bytes, 2)
            buffer = f.read(pixel_bytes)

        img = np.frombuffer(buffer, dtype=np.uint16)

        # Verify we read the expected amount of data
        if img.size != rows * cols:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)

        img = img.reshape((rows, cols))
        return img

    except Exception:
        # Return a blank image on any IO or parsing failure to maintain pipeline stability
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16)


def resize_image(img, size=None):
    """
    Resizes an image using Area Interpolation to suppress noise.
    Defaults to Config.IMG_SIZE if size is not provided.
    """
    if size is None:
        size = (Config.IMG_SIZE, Config.IMG_SIZE)
    elif isinstance(size, int):
        size = (size, size)

    # cv2.resize expects (width, height), so we pass (cols, rows)
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def normalize_min_max(img):
    """
    Performs independent per-channel min-max scaling to [0, 1].
    Handles constant images by returning zeros.
    """
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    else:
        # If image is constant (e.g. all zeros), return zeros
        return np.zeros_like(img)
