import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_raw(path):
    """
    Loads a DICOM file as a raw binary, infers resolution based on file size,
    strips the header, and returns a normalized numpy array.

    This function is a fallback for environments where pydicom or opencv
    cannot read the specific DICOM format. It assumes uncompressed 16-bit data.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: A 2D numpy array of shape (H, W) with values normalized to [0, 1].
                    Returns a zero array if loading fails.
    """
    if not os.path.exists(path):
        return np.zeros((256, 256), dtype=np.float32)

    try:
        # Read the entire file as bytes
        with open(path, "rb") as f:
            raw_data = f.read()

        file_size = len(raw_data)

        # Calculate expected byte counts for standard resolutions (16-bit = 2 bytes/pixel)
        bytes_512 = 512 * 512 * 2  # 524,288 bytes
        bytes_256 = 256 * 256 * 2  # 131,072 bytes

        # Infer shape based on file size
        # Files are slightly larger than pixel data due to headers
        if file_size >= bytes_512:
            shape = (512, 512)
            pixel_bytes_needed = bytes_512
        elif file_size >= bytes_256:
            shape = (256, 256)
            pixel_bytes_needed = bytes_256
        else:
            # File too small to contain expected image data
            return np.zeros((256, 256), dtype=np.float32)

        # Extract pixel data from the end of the file
        pixel_data = raw_data[-pixel_bytes_needed:]

        # Convert bytes to numpy array (Little Endian 16-bit Unsigned Integer)
        img = np.frombuffer(pixel_data, dtype=np.uint16).astype(np.float32)

        # Reshape to 2D image
        if img.size == shape[0] * shape[1]:
            img = img.reshape(shape)
        else:
            return np.zeros((256, 256), dtype=np.float32)

        # Min-Max Normalization
        img_min = np.min(img)
        img_max = np.max(img)

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            # Handle constant image (e.g., all black)
            img = np.zeros_like(img)

        return img

    except Exception:
        # Return blank image on any error to prevent pipeline crash
        return np.zeros((256, 256), dtype=np.float32)
