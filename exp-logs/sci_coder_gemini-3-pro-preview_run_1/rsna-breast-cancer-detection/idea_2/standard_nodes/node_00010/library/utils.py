import os
import random
import numpy as np
import torch
import cv2
import rasterio
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crop_breast_roi(image):
    """
    Crops the breast region of interest (ROI) by removing the black background.
    Assumes the background is approximately 0.
    """
    # Create a binary mask where pixels are greater than a low threshold
    # Using 0 is standard for DICOM padding, but >0 handles slight noise
    mask = image > 0

    # If the image is completely black, return original
    if mask.sum() == 0:
        return image

    # Find the bounding box of the non-zero region
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # Slice is exclusive at the end

    # Crop the image
    cropped_image = image[y0:y1, x0:x1]
    return cropped_image


def load_and_process_image(relative_path, load_cached_data=False):
    """
    Loads an image from the input directory, processes it (crop + resize),
    and implements a caching mechanism.

    Args:
        relative_path (str): Path relative to INPUT_DIR (e.g., 'train_images/123/456.dcm')
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: The processed image of shape (H, W) with values in [0, 255].
    """
    # Construct full input path
    full_path = os.path.join(Config.INPUT_DIR, relative_path)

    # Construct cache path
    # Flatten directory structure for cache filename: train_images/123/456.dcm -> train_images_123_456.npy
    safe_filename = relative_path.replace(os.sep, "_").replace(".", "_") + ".npy"
    cache_subdir = os.path.join(Config.CACHE_DIR, "processed_images")
    cache_path = os.path.join(cache_subdir, safe_filename)

    # 1. Try to load from cache if requested
    if load_cached_data:
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                # If loading fails (corrupt file), proceed to re-compute
                pass

    # 2. Compute: Load, Normalize, Crop, Resize
    img = None

    # Attempt 1: Rasterio (Good for JPEG 2000 DICOMs)
    try:
        with rasterio.open(full_path) as src:
            img = src.read(1)  # Read the first band
    except Exception:
        pass

    # Attempt 2: OpenCV (Fallback)
    if img is None:
        try:
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Handle Load Failure
    if img is None:
        # Return a blank image to prevent pipeline crash, but this indicates a data issue
        img = np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1]), dtype=np.uint8)
    else:
        # Normalize to 0-255 uint8
        if img.dtype != np.uint8:
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
                img = (img * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        # Crop ROI
        img = crop_breast_roi(img)

        # Resize to target size (width, height)
        img = cv2.resize(
            img,
            (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(cache_subdir, exist_ok=True)
    try:
        np.save(cache_path, img)
    except Exception:
        pass  # Non-critical failure if cache write fails

    return img
