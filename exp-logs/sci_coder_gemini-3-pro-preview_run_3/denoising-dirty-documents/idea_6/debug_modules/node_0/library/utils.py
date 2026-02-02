import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from library import config


def load_grayscale_image(path):
    """
    Loads an image from the specified path, converts it to grayscale,
    and normalizes pixel intensities to the range [0, 1].

    Args:
        path (str): The file path to the image.

    Returns:
        numpy.ndarray: The normalized grayscale image with shape (H, W) and dtype float32.
    """
    # Load image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")

    # Normalize pixel values to [0, 1]
    img_normalized = img.astype(np.float32) / 255.0

    return img_normalized


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both numpy arrays and torch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The RMSE value.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to treat them as a simple list of pixels
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    return float(rmse)


def format_submission(test_ids, predictions, output_path):
    """
    Formats the predictions into the required submission CSV format.

    Format:
    id,value
    110_1_1,0.5
    110_1_2,0.6
    ...

    Args:
        test_ids (list): List of image filenames (e.g., ['110.png', ...]).
        predictions (list of np.ndarray): List of predicted image arrays corresponding to test_ids.
        output_path (str): Path to save the submission CSV.
    """
    ids_list = []
    values_list = []

    print(f"Formatting submission for {len(test_ids)} images...")

    for filename, img in zip(test_ids, predictions):
        # Extract ID from filename (remove extension)
        # e.g., "110.png" -> "110"
        image_id = os.path.splitext(filename)[0]

        # Ensure image is a numpy array
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # Ensure image is in [0, 1] range
        img = np.clip(img, 0.0, 1.0)

        h, w = img.shape[:2]

        # Create grid of indices
        # Rows and Cols are 1-based as per task description (e.g., 1_2_1)
        rows, cols = np.indices((h, w))
        rows = rows + 1
        cols = cols + 1

        # Flatten everything
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        vals_flat = img.flatten()

        # Construct IDs: "{image_id}_{row}_{col}"
        # Using list comprehension
        current_ids = [f"{image_id}_{r}_{c}" for r, c in zip(rows_flat, cols_flat)]

        ids_list.extend(current_ids)
        values_list.extend(vals_flat)

    # Create DataFrame
    df_submission = pd.DataFrame({"id": ids_list, "value": values_list})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_submission)} rows.")
