import os
import random
import numpy as np
import torch
import cv2
from library.config import SEED, IMG_SIZE


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_robust(file_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Reads a DICOM file robustly.

    1. Attempts to read using OpenCV.
    2. If that fails, attempts a 'Raw Binary Tail-Read' assuming standard
       MRI dimensions (512x512 or 256x256) and 16-bit depth.
    3. Resizes the image to the target_size using Area Interpolation.

    Args:
        file_path (str): Path to the DICOM file.
        target_size (tuple): Desired (width, height) for the output image.

    Returns:
        np.ndarray: The image array (height, width). Returns a zero array if loading fails.
    """
    img = None

    # Attempt 1: OpenCV
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Attempt 2: Raw Binary Tail-Read
    if img is None:
        try:
            file_size = os.path.getsize(file_path)

            # Define expected sizes for uint16 images (2 bytes per pixel)
            size_512 = 512 * 512 * 2  # 524,288 bytes
            size_256 = 256 * 256 * 2  # 131,072 bytes

            with open(file_path, "rb") as f:
                if file_size >= size_512:
                    # Read last 512*512*2 bytes
                    f.seek(-size_512, os.SEEK_END)
                    data = f.read(size_512)
                    arr = np.frombuffer(data, dtype=np.uint16)
                    if arr.size == 512 * 512:
                        img = arr.reshape((512, 512))
                elif file_size >= size_256:
                    # Read last 256*256*2 bytes
                    f.seek(-size_256, os.SEEK_END)
                    data = f.read(size_256)
                    arr = np.frombuffer(data, dtype=np.uint16)
                    if arr.size == 256 * 256:
                        img = arr.reshape((256, 256))
        except Exception:
            img = None

    # Final Check and Resize
    if img is None:
        # Return zero array if all fails
        return np.zeros(target_size, dtype=np.float32)

    # Ensure image is resized to target dimensions
    # cv2.INTER_AREA is best for downsampling (denoising)
    try:
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    except Exception:
        return np.zeros(target_size, dtype=np.float32)

    return img
