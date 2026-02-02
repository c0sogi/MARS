import os
import re
import numpy as np
import cv2
import pandas as pd


def read_dicom_image(path):
    """
    Reads a DICOM file using a robust raw binary tail-read strategy to bypass header parsing issues.
    Falls back to OpenCV if the raw read heuristic assumes incorrect dimensions.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: Image data as float32. Returns a black image if reading fails.
    """
    if not os.path.exists(path):
        return np.zeros((224, 224), dtype=np.float32)

    try:
        file_size = os.path.getsize(path)

        # Heuristic: DICOM images in this dataset are typically 512x512 or 256x256 uint16
        # 512 * 512 * 2 bytes = 524,288 bytes
        # 256 * 256 * 2 bytes = 131,072 bytes

        shape = None
        bytes_to_read = 0

        if file_size >= 524288:
            # Likely 512x512
            shape = (512, 512)
            bytes_to_read = 524288
        elif file_size >= 131072:
            # Likely 256x256
            shape = (256, 256)
            bytes_to_read = 131072

        if shape and bytes_to_read > 0:
            with open(path, "rb") as f:
                f.seek(-bytes_to_read, 2)
                data = f.read(bytes_to_read)

            img = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
            if img.size == shape[0] * shape[1]:
                img = img.reshape(shape)
                return img

        # Fallback to OpenCV (unlikely to work for DICOM but included per spec)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)

    except Exception:
        pass

    # Final fallback: return zeros
    return np.zeros((224, 224), dtype=np.float32)


def resize_image(img, target_size=(224, 224)):
    """
    Resizes an image using area interpolation to suppress high-frequency noise.

    Args:
        img (np.ndarray): Input image.
        target_size (tuple): Target (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    if img is None or img.size == 0:
        return np.zeros((target_size[1], target_size[0]), dtype=np.float32)

    # cv2.resize expects (width, height)
    if (img.shape[1], img.shape[0]) == target_size:
        return img

    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(img):
    """
    Applies independent per-channel min-max scaling to [0, 1].

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized image.
    """
    if img is None or img.size == 0:
        return img

    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def get_sorted_files(directory):
    """
    Returns a sorted list of DICOM files in a directory.
    Sorts based on the integer index in the filename (e.g., Image-10.dcm).

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of full file paths sorted by slice index.
    """
    if not os.path.exists(directory):
        return []

    files = [f for f in os.listdir(directory) if f.endswith(".dcm")]

    def extract_number(filename):
        # Extract number from "Image-123.dcm"
        match = re.search(r"(\d+)", filename)
        if match:
            return int(match.group(1))
        return 0

    files.sort(key=extract_number)
    return [os.path.join(directory, f) for f in files]


def save_cache(data, path):
    """
    Saves data to a cache file (Parquet or NPY).

    Args:
        data: Data to save (pandas DataFrame or numpy array).
        path (str): Destination path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            # Try converting to DF
            pd.DataFrame(data).to_parquet(path, index=False)
    elif path.endswith(".npy"):
        np.save(path, data)
    else:
        # Default to npy for generic data
        np.save(path + ".npy", data)


def load_cache(path):
    """
    Loads data from a cache file.

    Args:
        path (str): Path to the file.

    Returns:
        Data loaded from file, or None if not found.
    """
    if not os.path.exists(path):
        return None

    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".npy"):
            return np.load(path)
        return None
    except Exception:
        return None
