import os
import cv2
import numpy as np
import pydicom
import torch
from library.config import Config


def read_dicom(path: str, fix_monochrome: bool = True):
    """
    Reads a DICOM file and returns the pixel array.
    Handles MONOCHROME1 inversion to ensure 0 is black (background).
    """
    if not os.path.exists(path):
        # Return None so the caller can handle it (e.g., return a blank tensor)
        return None

    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array

        # Handle Photometric Interpretation
        # MONOCHROME1: 0 is White, Max is Black.
        # MONOCHROME2: 0 is Black, Max is White.
        # We want 0=Black (Background), Max=White (Tissue).
        if fix_monochrome and hasattr(dicom, "PhotometricInterpretation"):
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = np.max(img) - img

        return img
    except Exception as e:
        print(f"Error reading DICOM {path}: {e}")
        return None


def get_roi_bbox(img: np.ndarray):
    """
    Calculates the bounding box of the breast tissue (Region of Interest).
    Assumes background is near 0.
    """
    # Normalize to 0-255 for thresholding calculation
    if img.max() > img.min():
        img_u8 = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_u8 = np.zeros_like(img, dtype=np.uint8)

    # Threshold to separate tissue from background
    # Config.ROI_BINARIZE_THRESHOLD is typically 0.05 (5%)
    thresh_val = int(255 * Config.ROI_BINARIZE_THRESHOLD)
    _, bin_img = cv2.threshold(img_u8, thresh_val, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No signal found, return full image dimensions
        return 0, img.shape[0], 0, img.shape[1]

    # Find largest contour by area (assumed to be the breast)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Return y_min, y_max, x_min, x_max
    return y, y + h, x, x + w


def process_image(path: str):
    """
    Full pipeline: Read -> ROI Crop -> Resize -> Percentile Norm -> RGB -> ImageNet Norm -> Tensor.
    Returns a torch Tensor (C, H, W).
    """
    # 1. Read
    img = read_dicom(path)

    if img is None:
        # Return a black tensor if read fails to prevent pipeline crash
        return torch.zeros(
            (3, Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=torch.float32
        )

    # 2. ROI Crop
    if Config.ROI_IGNORE_BLACK:
        y_min, y_max, x_min, x_max = get_roi_bbox(img)
        # Ensure crop is valid (has positive area)
        if (y_max > y_min) and (x_max > x_min):
            img = img[y_min:y_max, x_min:x_max]

    # 3. Resize
    # cv2.resize expects (width, height)
    img = cv2.resize(
        img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
    )

    # 4. Percentile Normalization (Idea 3 specific)
    # Convert to float for calculation
    img = img.astype(np.float32)

    # Clip to 1st and 99th percentile to remove outliers (hot pixels or artifacts)
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)

    if p99 > p1:
        img = np.clip(img, p1, p99)
        # Scale to [0, 1]
        img = (img - p1) / (p99 - p1)
    else:
        # If image is flat (e.g., all black), zero it out
        img = np.zeros_like(img)

    # 5. Convert to 3 Channels (RGB)
    # EfficientNet expects 3 input channels. We duplicate the grayscale channel.
    img = np.stack([img, img, img], axis=-1)  # (H, W, 3)

    # 6. ImageNet Normalization
    mean = np.array(Config.MEAN, dtype=np.float32)
    std = np.array(Config.STD, dtype=np.float32)
    img = (img - mean) / std

    # 7. Convert to Tensor (C, H, W)
    img = img.transpose(2, 0, 1)  # (3, H, W)
    return torch.tensor(img, dtype=torch.float32)
