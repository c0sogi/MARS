import os
import cv2
import numpy as np
import pandas as pd
from library.config import PIXEL_MAX


def load_normalized_image(path):
    """
    Loads an image from the specified path, converts it to grayscale,
    and normalizes pixel intensities to the range [0, 1].

    Args:
        path (str): The file path to the image.

    Returns:
        np.ndarray: The normalized image as a float32 numpy array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    # Load as grayscale
    # IMREAD_GRAYSCALE ensures we get a 2D array (H, W)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(
            f"Failed to load image at {path}. The file might be corrupted or format not supported."
        )

    # Normalize to [0, 1]
    img_normalized = img.astype(np.float32) / PIXEL_MAX

    return img_normalized


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The RMSE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)


def format_submission(predictions_dict, output_path):
    """
    Formats the predictions into the required submission format and saves to a CSV file.

    The format requires melting the image into pixels with id 'image_row_col'.
    Rows and columns are 1-indexed.

    Args:
        predictions_dict (dict): A dictionary where keys are image filenames (e.g., '110.png')
                                 and values are numpy arrays of the predicted image (H, W).
        output_path (str): The path to save the submission CSV.
    """
    ids = []
    values = []

    # Sort keys for deterministic output order
    sorted_filenames = sorted(predictions_dict.keys())

    for filename in sorted_filenames:
        img_pred = predictions_dict[filename]

        # Extract image ID from filename (e.g., '110.png' -> '110')
        # We assume the filename in the dict keys matches the metadata 'image_id' format
        image_id_str = os.path.splitext(filename)[0]

        height, width = img_pred.shape

        # Generate coordinate grids (0-based)
        grid_r, grid_c = np.indices((height, width))

        # Convert to 1-based indexing as per task description and sample submission
        grid_r = grid_r + 1
        grid_c = grid_c + 1

        # Flatten arrays to 1D lists of pixels
        flat_r = grid_r.flatten()
        flat_c = grid_c.flatten()
        flat_val = img_pred.flatten()

        # Create ID strings: "image_row_col"
        # Using list comprehension is efficient enough for the dataset size
        chunk_ids = [f"{image_id_str}_{r}_{c}" for r, c in zip(flat_r, flat_c)]

        ids.extend(chunk_ids)
        values.extend(flat_val)

    # Create DataFrame
    df_submission = pd.DataFrame({"id": ids, "value": values})

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV without the index
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_submission)} rows.")
