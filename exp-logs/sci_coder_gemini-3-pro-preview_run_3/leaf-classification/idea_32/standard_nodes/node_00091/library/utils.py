import os
import random
import numpy as np
import cv2
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(path, target_size=None):
    """
    Loads an image from the specified path using OpenCV.

    Args:
        path (str): Path to the image file.
        target_size (tuple, optional): A tuple (width, height) to resize the image.
                                       If None, the original resolution is returned.

    Returns:
        np.ndarray: The loaded image array (grayscale).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    # Load as grayscale since the dataset is binary (black leaf on white bg)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load image at: {path}. The file might be corrupt.")

    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


def rotate_image(image, angle, border_value=255):
    """
    Rotates an image by a specific angle around its center.

    Args:
        image (np.ndarray): The input image array.
        angle (float): The angle of rotation in degrees (counter-clockwise).
        border_value (int): The value to fill the newly created border area.
                            Defaults to 255 (white) for the leaf dataset.

    Returns:
        np.ndarray: The rotated image with the same dimensions as the input.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the affine transformation
    # INTER_LINEAR is used for smoother edges, which is better for CNN features
    # BORDER_CONSTANT with value 255 ensures the background remains white
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated


def clip_probabilities(probs, eps=Config.PROB_CLIP_EPS):
    """
    Clips probability values to the range [eps, 1-eps] to avoid extremes in log loss calculation.

    Args:
        probs (np.ndarray): Array of probability values.
        eps (float): The epsilon value for clipping. Defaults to Config.PROB_CLIP_EPS.

    Returns:
        np.ndarray: The clipped probability array.
    """
    return np.maximum(np.minimum(probs, 1 - eps), eps)
