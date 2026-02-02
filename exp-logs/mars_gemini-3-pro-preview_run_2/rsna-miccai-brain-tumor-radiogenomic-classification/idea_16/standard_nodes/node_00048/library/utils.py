import os
import numpy as np
import cv2
from library.config import set_seed, IMG_SIZE, SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility.
    Wrapper around library.config.set_seed.
    """
    set_seed(seed)


def read_dicom_robust(file_path):
    """
    Reads a DICOM file using a hierarchical strategy (Cite Lesson 39).
    1. Attempt standard read with OpenCV.
    2. Fallback to raw binary tail-read if standard read fails.

    Args:
        file_path (str): Path to the .dcm file.

    Returns:
        np.ndarray: Image array of shape (IMG_SIZE, IMG_SIZE) with dtype float32.
    """
    # 1. Primary Method: OpenCV (Cite Lesson 39)
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Handle dimensions (some DICOMs might be read as 3-channel by opencv)
            if img.ndim == 3:
                img = img[:, :, 0]

            img = img.astype(np.float32)

            # Resize using Area Interpolation (Cite Lesson 31)
            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = cv2.resize(
                    img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )

            return img
    except Exception:
        pass

    # 2. Fallback Method: Raw Binary Tail-Read (Cite Lesson 32)
    try:
        file_size = os.path.getsize(file_path)

        # Define byte sizes for common DICOM resolutions (16-bit depth)
        SIZE_512 = 512 * 512 * 2
        SIZE_256 = 256 * 256 * 2

        # Heuristic to determine dimensions based on file size
        if file_size >= SIZE_512:
            rows, cols = 512, 512
            num_bytes = SIZE_512
        elif file_size >= SIZE_256:
            rows, cols = 256, 256
            num_bytes = SIZE_256
        else:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        with open(file_path, "rb") as f:
            f.seek(-num_bytes, 2)
            raw_data = f.read(num_bytes)

        img = np.frombuffer(raw_data, dtype=np.uint16)
        img = img.reshape((rows, cols))
        img = img.astype(np.float32)

        if rows != IMG_SIZE or cols != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        return img

    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
