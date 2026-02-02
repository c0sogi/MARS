import os
import random
import numpy as np
import cv2
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from the specified path.

    Args:
        path (str): Full path to the image file.

    Returns:
        np.ndarray: The loaded image in RGB format (H, W, 3) with uint8 dtype.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    # Read image in color mode (default BGR in OpenCV)
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to read image at: {path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image by a specific angle around its center.
    The background is filled with white (255) to match the leaf dataset characteristics.

    Args:
        image (np.ndarray): Input image array (H, W, C).
        angle (float): Rotation angle in degrees.

    Returns:
        np.ndarray: The rotated image with the same dimensions as the input.
    """
    if angle == 0:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform affine transformation
    # Use borderValue=(255, 255, 255) for white background to match the dataset
    rotated = cv2.warpAffine(
        image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )

    return rotated
