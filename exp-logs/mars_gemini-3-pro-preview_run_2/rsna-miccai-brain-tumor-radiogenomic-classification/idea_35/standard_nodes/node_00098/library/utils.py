import os
import random
import numpy as np
import torch
import cv2


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_raw(path, img_size=224):
    """
    Reads a DICOM file using raw binary tail-read logic to bypass header parsing.
    Assumes 16-bit depth and standard resolutions (512x512 or 256x256).

    Args:
        path (str): Path to the .dcm file.
        img_size (int): Target spatial dimension (square).

    Returns:
        np.ndarray: Float32 image array of shape (img_size, img_size).
                    Returns a zero-filled array if loading fails.
    """
    try:
        if not os.path.exists(path):
            return np.zeros((img_size, img_size), dtype=np.float32)

        file_size = os.path.getsize(path)

        # Heuristic to determine resolution based on file size
        # 512x512 @ 16-bit = 524,288 bytes
        # 256x256 @ 16-bit = 131,072 bytes
        # We check if the file is large enough to contain these raw buffers.

        if file_size >= 524288:
            res = 512
            num_bytes = 524288
        elif file_size >= 131072:
            res = 256
            num_bytes = 131072
        else:
            # File too small for known resolutions
            return np.zeros((img_size, img_size), dtype=np.float32)

        # Read the last num_bytes from the file (Tail-Read)
        with open(path, "rb") as f:
            # Seek to the start of the pixel data (assuming it's at the end)
            f.seek(file_size - num_bytes)
            data = f.read(num_bytes)

        # Convert binary data to numpy array
        # MRI data is typically stored as uint16
        img = np.frombuffer(data, dtype=np.uint16).astype(np.float32)

        # Verify shape consistency
        if img.size != res * res:
            return np.zeros((img_size, img_size), dtype=np.float32)

        img = img.reshape((res, res))

        # Resize to target size using Area interpolation (best for downsampling)
        if res != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img

    except Exception:
        # Return black image on any failure to ensure pipeline stability
        return np.zeros((img_size, img_size), dtype=np.float32)
