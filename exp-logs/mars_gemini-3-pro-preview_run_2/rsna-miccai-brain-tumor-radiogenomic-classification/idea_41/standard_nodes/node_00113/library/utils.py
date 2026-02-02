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
    2. If that fails, attempts a 'Raw Binary Tail-Read' checking multiple common
       MRI dimensions (512, 256, 240, 192) to handle missing codecs.
    3. Resizes the image to the target_size using Area Interpolation.

    Cite solution_lesson_node_00107: Robust Data Ingestion
    Cite solution_lesson_node_00101: Standard Libraries vs Custom Heuristics (Tiered approach)

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

            # Common MRI dimensions to check (descending order)
            # 512x512, 256x256, 240x240 (BraTS common), 192x192
            dims_to_check = [512, 256, 240, 192]

            with open(file_path, "rb") as f:
                for dim in dims_to_check:
                    num_bytes = dim * dim * 2  # 16-bit
                    if file_size >= num_bytes:
                        try:
                            f.seek(-num_bytes, os.SEEK_END)
                            data = f.read(num_bytes)
                            arr = np.frombuffer(data, dtype=np.uint16)
                            if arr.size == dim * dim:
                                img = arr.reshape((dim, dim))
                                break
                        except Exception:
                            continue
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
