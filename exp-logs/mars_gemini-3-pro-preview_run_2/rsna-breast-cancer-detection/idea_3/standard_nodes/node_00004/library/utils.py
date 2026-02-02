import os
import cv2
import numpy as np
import torch
import random
from library.config import Config


def seed_everything(seed=42):
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


def probabilistic_f1(y_true, y_pred, beta=1):
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities (0 to 1).
        beta: Weight of recall in the F-score (default 1).

    Returns:
        float: The probabilistic F1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate probabilistic True Positives and False Positives
    pTP = np.sum(y_true * y_pred)
    pFP = np.sum((1 - y_true) * y_pred)

    # Total Positives (TP + FN) is simply the sum of positive labels
    total_positives = np.sum(y_true)

    epsilon = 1e-7

    # Calculate probabilistic Precision and Recall
    pPrecision = pTP / (pTP + pFP + epsilon)
    pRecall = pTP / (total_positives + epsilon)

    # Calculate F1
    f1 = (
        (1 + beta**2)
        * (pPrecision * pRecall)
        / ((beta**2 * pPrecision) + pRecall + epsilon)
    )

    return f1


def read_dicom_bytes(path, fix_monochrome=True):
    """
    Reads a DICOM file as a binary stream, searches for image headers
    (JPEG/JPEG2000), and decodes the payload using OpenCV.
    Handles 16-bit to 8-bit conversion and photometric inversion.

    Args:
        path: Path to the .dcm file.
        fix_monochrome: If True, checks if the image background is white and inverts it.

    Returns:
        np.ndarray: The image as a numpy array (H, W, 3) in RGB format.
    """
    # Return blank image if file doesn't exist
    if not os.path.exists(path):
        return np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8)

    with open(path, "rb") as f:
        data = f.read()

    img = None

    # Binary Signatures
    jpeg_start = b"\xff\xd8"
    j2k_codestream_start = b"\xff\x4f\xff\x51"
    j2k_file_start = b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a"

    # 1. Attempt to find and decode JPEG 2000 Codestream (common in DICOM)
    start_idx = data.find(j2k_codestream_start)
    if start_idx != -1:
        try:
            # Decode from the found header to the end of the buffer
            img = cv2.imdecode(
                np.frombuffer(data[start_idx:], np.uint8), cv2.IMREAD_UNCHANGED
            )
        except Exception:
            pass

    # 2. Attempt to find and decode JPEG 2000 File Format
    if img is None:
        start_idx = data.find(j2k_file_start)
        if start_idx != -1:
            try:
                img = cv2.imdecode(
                    np.frombuffer(data[start_idx:], np.uint8), cv2.IMREAD_UNCHANGED
                )
            except Exception:
                pass

    # 3. Attempt to find and decode JPEG
    # DICOM files might contain thumbnails. We search for all JPEG headers and pick the largest valid image.
    if img is None:
        indices = []
        pos = 0
        while True:
            idx = data.find(jpeg_start, pos)
            if idx == -1:
                break
            indices.append(idx)
            pos = idx + 2

        best_size = 0
        for idx in indices:
            try:
                curr = cv2.imdecode(
                    np.frombuffer(data[idx:], np.uint8), cv2.IMREAD_UNCHANGED
                )
                if curr is not None:
                    if curr.size > best_size:
                        best_size = curr.size
                        img = curr
            except Exception:
                continue

    # Fallback: Return black image if decoding completely failed
    if img is None:
        return np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8)

    # 4. Handle Bit Depth (Normalize 16-bit to 8-bit)
    if img.dtype == np.uint16:
        img = (img / 65535.0 * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        # Normalize float or other types to 0-255
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # 5. Fix Monochrome (Invert if background is white)
    if fix_monochrome:
        h, w = img.shape[:2]
        # Check corners to determine background intensity
        if h > 20 and w > 20:
            tl = img[0:20, 0:20].mean()
            tr = img[0:20, w - 20 : w].mean()
            bl = img[h - 20 : h, 0:20].mean()
            br = img[h - 20 : h, w - 20 : w].mean()
            mean_corner = (tl + tr + bl + br) / 4.0

            # If corners are bright (>127), assume inverted photometric interpretation
            if mean_corner > 127:
                img = 255 - img

    # 6. Ensure Output is RGB (H, W, 3)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # OpenCV loads as BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
