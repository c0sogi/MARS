import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import mean_squared_error
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(path):
    """
    Loads an image from the specified path in grayscale and normalizes it to [0, 1].

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Normalized grayscale image as float32 array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    # Read image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load image from {path}")

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error between two images or arrays.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The RMSE value.
    """
    # Ensure inputs are flattened numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    return rmse


def save_submission(predictions, submission_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the required submission format and saves to CSV.

    Args:
        predictions (dict): Dictionary mapping image_id (str) to predicted image array (numpy array).
                            image_id should be the filename (e.g., "110.png" or "110").
        submission_path (str): Path to save the CSV file.
    """
    # List to store dataframes for each image
    dfs = []

    # Sort keys to ensure deterministic order
    sorted_ids = sorted(predictions.keys())

    for img_id in sorted_ids:
        img_arr = predictions[img_id]

        # Ensure values are within [0, 1]
        img_arr = np.clip(img_arr, 0, 1)

        h, w = img_arr.shape

        # Create grid of indices (1-based)
        # rows correspond to y-axis (height), cols to x-axis (width)
        grid = np.indices((h, w))
        rows = grid[0] + 1
        cols = grid[1] + 1

        # Flatten arrays
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        vals_flat = img_arr.flatten()

        # Create temporary DataFrame
        df_img = pd.DataFrame({"row": rows_flat, "col": cols_flat, "value": vals_flat})

        # Clean ID: remove extension if present (e.g., "110.png" -> "110")
        clean_img_id = os.path.splitext(img_id)[0]

        # Construct ID column: {clean_img_id}_{row}_{col}
        df_img["id"] = (
            f"{clean_img_id}_"
            + df_img["row"].astype(str)
            + "_"
            + df_img["col"].astype(str)
        )

        # Select only required columns
        df_img = df_img[["id", "value"]]
        dfs.append(df_img)

    # Concatenate all
    if not dfs:
        return

    final_df = pd.concat(dfs, ignore_index=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Save
    final_df.to_csv(submission_path, index=False)
