import os
import random
import numpy as np
import torch
import cv2
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dicom_image(file_path, target_size=(config.IMG_SIZE, config.IMG_SIZE)):
    """
    Loads a DICOM image with a robust fallback strategy.

    1. Attempts to load using OpenCV.
    2. If that fails, falls back to reading raw binary pixel data from the end of the file.
       It infers dimensions (512x512 or 256x256) based on file size.

    Args:
        file_path (str): Path to the DICOM file.
        target_size (tuple): Desired output size (width, height).

    Returns:
        np.ndarray: The image array in float32 format, resized to target_size.
    """
    img = None

    # Strategy 1: Try OpenCV
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Strategy 2: Raw Binary Tail-Read Fallback
    if img is None:
        try:
            file_size = os.path.getsize(file_path)

            # Constants for 16-bit images
            # 512x512 pixels * 2 bytes = 524,288 bytes
            # 256x256 pixels * 2 bytes = 131,072 bytes
            SIZE_512 = 512 * 512 * 2
            SIZE_256 = 256 * 256 * 2

            raw_shape = None
            read_size = 0

            if file_size >= SIZE_512:
                raw_shape = (512, 512)
                read_size = SIZE_512
            elif file_size >= SIZE_256:
                raw_shape = (256, 256)
                read_size = SIZE_256

            if raw_shape is not None:
                with open(file_path, "rb") as f:
                    # Seek to the start of the pixel data (end of file - image bytes)
                    f.seek(file_size - read_size)
                    raw_data = f.read(read_size)

                # Convert binary data to numpy array
                img_array = np.frombuffer(raw_data, dtype=np.uint16)

                # Reshape if the element count matches
                if img_array.size == raw_shape[0] * raw_shape[1]:
                    img = img_array.reshape(raw_shape)
        except Exception:
            # If fallback fails, img remains None
            pass

    # Safety: If all loading failed, return a black image
    if img is None:
        img = np.zeros(target_size, dtype=np.float32)
    else:
        # Ensure image is float32
        img = img.astype(np.float32)

        # Resize using Area Interpolation (best for downsampling/preserving structure)
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img
