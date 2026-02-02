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
    Reads a DICOM file using a raw binary tail-read approach.
    This bypasses potential header corruption and dependency on pydicom.

    Logic:
    1. Check file size to determine resolution (512x512 or 256x256).
    2. Read the last N bytes corresponding to the pixel data.
    3. Convert to float32 to preserve contrast.
    4. Resize to IMG_SIZE x IMG_SIZE using Area Interpolation (low-pass filter).

    Args:
        file_path (str): Path to the .dcm file.

    Returns:
        np.ndarray: Image array of shape (IMG_SIZE, IMG_SIZE) with dtype float32.
                    Returns a zero-filled array if reading fails.
    """
    try:
        file_size = os.path.getsize(file_path)

        # Define byte sizes for common DICOM resolutions (16-bit depth)
        # 512 * 512 * 2 = 524,288 bytes
        # 256 * 256 * 2 = 131,072 bytes
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
            # File too small to contain expected pixel data, return empty image
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        with open(file_path, "rb") as f:
            # Seek to the end minus the pixel data size (Tail-Read)
            f.seek(-num_bytes, 2)
            raw_data = f.read(num_bytes)

        # Load as uint16 (standard MRI bit depth)
        img = np.frombuffer(raw_data, dtype=np.uint16)

        # Reshape to original dimensions
        img = img.reshape((rows, cols))

        # Convert to float32
        img = img.astype(np.float32)

        # Resize to target size using Area Interpolation
        if rows != IMG_SIZE or cols != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        return img

    except Exception:
        # In case of any read error (IO, buffer size mismatch), return a zero array
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
