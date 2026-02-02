import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The RMSE value.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure element-wise comparison regardless of shape (H, W) vs (1, H, W)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    return np.sqrt(mse)


def normalize_image(image):
    """
    Normalizes a grayscale image from [0, 255] to [0, 1].

    Args:
        image (np.ndarray): Input image array (uint8 or float).

    Returns:
        np.ndarray: Normalized image array (float32).
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image):
    """
    Denormalizes an image from [0, 1] to [0, 255].

    Args:
        image (np.ndarray): Input image array (float).

    Returns:
        np.ndarray: Denormalized image array (uint8).
    """
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def load_image(image_path, cache_path=None, load_cached=True):
    """
    Loads an image from disk. Implements caching using .npy files to speed up access.

    Args:
        image_path (str): Path to the source image file.
        cache_path (str, optional): Path to the .npy cache file.
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The image array (grayscale).
    """
    # 1. Try to load from cache
    if load_cached and cache_path is not None and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            # If load fails, fall back to reading source
            pass

    # 2. Load from source
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")

    # 3. Save to cache if path provided
    if cache_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, img)

    return img


def create_submission(predictions, output_path):
    """
    Generates the submission CSV file from a dictionary of predictions.

    Args:
        predictions (dict): Dictionary mapping image IDs (str) to predicted image arrays (np.ndarray).
                            Arrays should be 2D (H, W) with values in [0, 1].
        output_path (str): Path to save the CSV file.
    """
    ids = []
    values = []

    # Sort IDs for deterministic output order
    sorted_ids = sorted(predictions.keys())

    for img_id in sorted_ids:
        pred_img = predictions[img_id]

        # Ensure 2D
        if pred_img.ndim == 3:
            pred_img = pred_img.squeeze()

        h, w = pred_img.shape

        # Vectorized generation of row and column indices (1-based)
        # We flatten in row-major order: (1,1), (1,2), ..., (1,W), (2,1)...

        # Row indices: [1, 1, ..., 2, 2, ...]
        rows = np.repeat(np.arange(1, h + 1), w)

        # Col indices: [1, 2, ..., 1, 2, ...]
        cols = np.tile(np.arange(1, w + 1), h)

        # Flatten pixel values
        flat_pixels = pred_img.flatten()

        # Generate ID strings: "{img_id}_{row}_{col}"
        # List comprehension is efficient enough for this operation
        pixel_ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

        ids.extend(pixel_ids)
        values.extend(flat_pixels)

    # Create DataFrame and save
    df = pd.DataFrame({"id": ids, "value": values})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
