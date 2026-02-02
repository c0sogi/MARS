import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from library import config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def pad_image_to_multiple(image, multiple=8):
    """
    Pads a numpy image (H, W) or (H, W, C) using reflection padding
    so that dimensions are divisible by 'multiple'.

    Args:
        image (np.ndarray): Input image.
        multiple (int): The divisor to pad to (default 8).

    Returns:
        tuple: (padded_image, pads) where pads is (top, bottom, left, right).
    """
    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3:
        h, w, c = image.shape
    else:
        raise ValueError("Image must be 2D or 3D")

    # Calculate required padding
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # Pad to bottom and right
    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    # Apply reflection padding
    # cv2.BORDER_REFLECT_101 is the default and standard reflection (gfedcb|abcdefgh|gfedcb)
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )

    return padded_image, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_image(padded_image, original_shape):
    """
    Crops the padded image back to the original shape.

    Args:
        padded_image (np.ndarray): The padded image.
        original_shape (tuple): The target shape (H, W) or (H, W, C).

    Returns:
        np.ndarray: The unpadded image.
    """
    h, w = original_shape[:2]
    return padded_image[:h, :w]


def calculate_rmse(y_true, y_pred):
    """
    Calculates Root Mean Squared Error between two numpy arrays.

    Args:
        y_true (np.ndarray): Ground truth.
        y_pred (np.ndarray): Predictions.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def create_submission(predictions, output_path):
    """
    Generates the submission CSV file from a dictionary of predictions.

    Args:
        predictions (dict): Dictionary {img_id (str): image_array (numpy array)}.
                            image_array should be normalized [0, 1].
        output_path (str): Path to save the CSV.
    """
    data_ids = []
    data_values = []

    # Sort keys for deterministic output order
    for img_id in sorted(predictions.keys()):
        pred = predictions[img_id]

        # Ensure 2D for processing
        if pred.ndim == 3:
            # If (H, W, 1), squeeze to (H, W)
            if pred.shape[2] == 1:
                pred = pred[:, :, 0]
            # If (H, W, C) with C > 1, we assume it's already processed or take channel 0
            # Task specifies grayscale, so this handles the common case.

        h, w = pred.shape[:2]

        # Create coordinate grids (1-based indexing)
        # rows: 1..h, cols: 1..w
        # np.repeat([1, 2], 2) -> [1, 1, 2, 2] (Rows stay same across columns)
        rows = np.repeat(np.arange(1, h + 1), w)
        # np.tile([1, 2], 2) -> [1, 2, 1, 2] (Cols cycle)
        cols = np.tile(np.arange(1, w + 1), h)

        # Flatten values (row-major)
        vals = pred.flatten()

        # Generate IDs: "image_row_col"
        # Using list comprehension
        current_ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

        data_ids.extend(current_ids)
        data_values.extend(vals)

    df = pd.DataFrame({"id": data_ids, "value": data_values})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
