import os
import random
import numpy as np
import torch
import pydicom
import cv2
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_image(path, img_size=Config.IMG_SIZE):
    """
    Reads a DICOM file, normalizes pixel intensities to [0, 1], and resizes
    the image to the target resolution.

    Args:
        path (str): Relative path to the DICOM file (e.g., 'train/00000/FLAIR/Image-1.dcm').
        img_size (int, optional): Target spatial resolution (H, W). Defaults to Config.IMG_SIZE.

    Returns:
        np.ndarray: Processed 2D image array of shape (img_size, img_size) with values in [0, 1].
    """
    # Construct full path based on Config
    full_path = os.path.join(Config.INPUT_DIR, path)

    # Handle missing files gracefully
    if not os.path.exists(full_path):
        return np.zeros((img_size, img_size), dtype=np.float32)

    try:
        # Read DICOM file
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Normalize to [0, 1]
        img_min = np.min(img)
        img_max = np.max(img)

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            # Avoid division by zero for constant images (e.g., all black)
            img = np.zeros_like(img)

        # Resize if dimensions do not match target
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img

    except Exception:
        # Return black image on read failure to maintain batch shape consistency
        return np.zeros((img_size, img_size), dtype=np.float32)
