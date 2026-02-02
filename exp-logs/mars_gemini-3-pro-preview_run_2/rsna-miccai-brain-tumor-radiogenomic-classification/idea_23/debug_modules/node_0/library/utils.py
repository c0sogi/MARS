import os
import random
import numpy as np
import torch
import cv2
import pydicom
from library.config import Config


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_robust(path):
    """
    Reads a DICOM file robustly, bypassing strict header checks.

    This implements the 'Raw Binary Tail-Read' strategy by using pydicom with
    force=True, which attempts to read the file even if the standard DICOM
    preamble is missing or the header is slightly malformed.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: The pixel array as float32. Returns a zero-filled array
                    if reading fails.
    """
    try:
        # force=True allows reading files with missing/corrupt DICOM headers
        dcm = pydicom.dcmread(path, force=True)

        # Extract pixel array and convert to float32 for precision
        img = dcm.pixel_array.astype(np.float32)

        return img
    except Exception:
        # Fallback: return a black image to prevent pipeline crash.
        # We use Config.IMG_SIZE as a default, though actual dimensions may vary.
        # This will be resized by the pipeline anyway.
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def normalize_channel(img):
    """
    Normalizes a single channel image to [0, 1] using independent min-max scaling.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized image array.
    """
    min_val = np.min(img)
    max_val = np.max(img)

    # Avoid division by zero if the image is constant (e.g., all black)
    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(img)


def get_downsampled_max_anchor(flair_path):
    """
    Determines the best anchor slice index for a subject based on the FLAIR modality.

    This implements the 'Robust Downsampled-Max' strategy:
    1. Sorts slices numerically.
    2. Restricts search to the 15%-85% depth range.
    3. Downsamples slices to 32x32 (Area Interpolation) to filter high-freq noise.
    4. Selects the slice with the highest max intensity in the downsampled domain.

    Args:
        flair_path (str): Path to the subject's FLAIR directory.

    Returns:
        int: The index of the anchor slice (relative to the sorted file list).
    """
    if not os.path.exists(flair_path):
        return 0

    # List files and filter for .dcm
    files = [f for f in os.listdir(flair_path) if f.endswith(".dcm")]

    if not files:
        return 0

    # Sort files numerically based on the integer in "Image-X.dcm"
    # This ensures we process the volume in the correct spatial order
    try:
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    except Exception:
        # Fallback to string sort if naming convention differs
        files.sort()

    num_slices = len(files)

    # Define search bounds (15% to 85% of volume depth)
    start_idx = int(num_slices * Config.ROI_DEPTH_MIN)
    end_idx = int(num_slices * Config.ROI_DEPTH_MAX)

    # Clamp bounds
    start_idx = max(0, start_idx)
    end_idx = min(num_slices, end_idx)

    # Handle edge case where range is empty
    if start_idx >= end_idx:
        return num_slices // 2

    max_intensity = -float("inf")
    best_index = start_idx

    # Iterate through the restricted depth range
    for i in range(start_idx, end_idx):
        file_path = os.path.join(flair_path, files[i])

        # Read image robustly
        img = read_dicom_robust(file_path)

        # Downsample to 32x32 using Area Interpolation
        # This acts as a spatial low-pass filter, suppressing single-pixel artifacts
        # while preserving larger hyperintense regions (tumors).
        try:
            img_small = cv2.resize(
                img,
                (Config.ROI_DOWNSAMPLE_SIZE, Config.ROI_DOWNSAMPLE_SIZE),
                interpolation=cv2.INTER_AREA,
            )

            # Calculate max intensity of the region
            current_max = np.max(img_small)

            if current_max > max_intensity:
                max_intensity = current_max
                best_index = i

        except Exception:
            # Skip slice if resizing fails (e.g., zero dimension)
            continue

    return best_index
