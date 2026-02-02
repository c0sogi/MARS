import os
import cv2
import numpy as np
import torch
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the set_seed function from library.config.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def load_and_preprocess_image(path, height=224, width=224, normalize=True):
    """
    Loads an image from disk, handles channel conversions, resizes, and converts to a PyTorch tensor.

    Args:
        path (str): Path to the image file.
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        normalize (bool): If True, scales pixel values to [0, 1].

    Returns:
        torch.Tensor: The processed image tensor with shape (C, H, W).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    # Load image using OpenCV
    # IMREAD_UNCHANGED allows us to detect alpha channels or grayscale
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load image at {path}. The file may be corrupt.")

    # Handle Channels
    if len(img.shape) == 2:
        # Grayscale (H, W) -> RGB (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Check channel count
        c = img.shape[2]
        if c == 3:
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif c == 4:
            # BGRA -> RGB (drop alpha)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            # Fallback for unusual channel counts, force conversion to RGB if possible
            # or just take first 3 channels if > 3
            if c > 3:
                img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # If < 3 (e.g. 1 but shaped as (H,W,1)), convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    if normalize:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    # Convert to Tensor format: (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))

    # Convert to PyTorch Tensor
    tensor = torch.from_numpy(img)

    return tensor
