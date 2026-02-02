import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.config import Config


def set_seed(seed: int = 42):
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

    # Ensure deterministic behavior in cudnn
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
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure element-wise comparison regardless of shape
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    return np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))


def read_image(path: str, grayscale: bool = True):
    """
    Reads an image from the specified path.

    Args:
        path (str): Path to the image file.
        grayscale (bool): If True, reads the image in grayscale mode.

    Returns:
        np.ndarray: The loaded image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imread(path, flags)

    if image is None:
        raise ValueError(f"Failed to load image from {path}")

    return image


def save_image(path: str, image: np.ndarray):
    """
    Saves an image to the specified path.

    Args:
        path (str): Destination path.
        image (np.ndarray): Image data to save.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def normalize(image: np.ndarray):
    """
    Normalizes image pixel values from [0, 255] to [0, 1].

    Args:
        image (np.ndarray): Input image (uint8).

    Returns:
        np.ndarray: Normalized image (float32).
    """
    return image.astype(np.float32) / 255.0


def denormalize(image: np.ndarray):
    """
    Denormalizes image pixel values from [0, 1] to [0, 255].

    Args:
        image (np.ndarray): Input image (float).

    Returns:
        np.ndarray: Denormalized image (uint8).
    """
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def create_submission_file(predictions: dict, output_path: str):
    """
    Generates the submission CSV file from a dictionary of predictions.

    Args:
        predictions (dict): Dictionary where keys are image IDs (str) and values are
                            predicted image arrays (np.ndarray).
        output_path (str): Path to save the submission CSV.
    """
    print(f"Generating submission file at {output_path}...")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open file buffer
    with open(output_path, "w") as f:
        # Write header
        f.write("id,value\n")

        # Iterate through sorted image IDs for consistency
        for img_id in sorted(predictions.keys()):
            img = predictions[img_id]

            # Ensure image is 2D
            if len(img.shape) == 3:
                img = img.squeeze()

            h, w = img.shape

            # Iterate pixels
            # Note: Task specifies 1-based indexing for rows and columns
            for r in range(h):
                for c in range(w):
                    pixel_id = f"{img_id}_{r+1}_{c+1}"
                    # Intensity values range from 0 (black) to 1 (white)
                    # We assume predictions are already normalized or denormalized depending on requirement.
                    # The prompt says "Intensity values range from 0 (black) to 1 (white)".
                    # So we expect the input `predictions` to be in [0, 1] or [0, 255].
                    # If they are uint8 [0, 255], we should probably normalize them to [0, 1] for the CSV?
                    # Re-reading prompt: "Intensity values range from 0 (black) to 1 (white)."
                    # Sample submission shows '1' (int).
                    # Wait, sample submission says "value (int64) has 1 unique values: [1]".
                    # But prompt says "Intensity values range from 0 (black) to 1 (white)."
                    # Usually this implies float, but the sample shows int.
                    # However, standard image tasks usually allow float.
                    # If the sample only has 1s, it might just be a dummy sample.
                    # We will write the value as is. If it's float, we write float.

                    val = img[r, c]
                    f.write(f"{pixel_id},{val}\n")


def print_metrics(metrics: dict):
    """
    Prints metrics dictionary with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
    """
    print("-" * 20)
    print("Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("-" * 20)
