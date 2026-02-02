import os
import random
import numpy as np
import cv2
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets random seeds for python, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def read_dicom_robust(path):
    """
    Reads a DICOM file. Attempts to use OpenCV first.
    If that fails, falls back to raw binary tail-read based on common MRI dimensions.
    Returns a float32 numpy array.
    """
    if not os.path.exists(path):
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Attempt 1: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = img.astype(np.float32)
            # Ensure 2D (H, W)
            if img.ndim == 3:
                img = img[:, :, 0]
            return img
    except Exception:
        pass

    # Attempt 2: Raw Binary Tail-Read
    # Common dimensions for this dataset based on file sizes
    # 512x512 uint16 = 524,288 bytes (File size ~525kB)
    # 256x256 uint16 = 131,072 bytes (File size ~132kB)
    try:
        file_size = os.path.getsize(path)

        shape = None
        num_pixels = 0

        if file_size >= 524288:
            shape = (512, 512)
            num_pixels = 512 * 512
        elif file_size >= 131072:
            shape = (256, 256)
            num_pixels = 256 * 256

        if shape is not None:
            with open(path, "rb") as f:
                # Seek to the end minus the pixel data size
                # uint16 = 2 bytes per pixel
                bytes_to_read = num_pixels * 2
                f.seek(-bytes_to_read, os.SEEK_END)
                data = f.read(bytes_to_read)

                img_array = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
                if img_array.size == num_pixels:
                    img = img_array.reshape(shape)
                    return img
    except Exception:
        pass

    # Fallback: Return zeros
    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def resize_image(img, size=(Config.IMG_SIZE, Config.IMG_SIZE)):
    """
    Resizes an image to the specified size using Area Interpolation
    to suppress high-frequency noise.
    """
    if img is None:
        return np.zeros(size, dtype=np.float32)

    try:
        # Ensure size is a tuple (width, height)
        if isinstance(size, int):
            target_size = (size, size)
        else:
            target_size = size

        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return np.zeros(size, dtype=np.float32)


def normalize_image(img):
    """
    Applies Min-Max scaling to [0, 1].
    Handles cases where max == min (constant image) by returning zeros.
    """
    if img is None:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    img_min = np.min(img)
    img_max = np.max(img)

    if img_max - img_min > 1e-6:
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)
