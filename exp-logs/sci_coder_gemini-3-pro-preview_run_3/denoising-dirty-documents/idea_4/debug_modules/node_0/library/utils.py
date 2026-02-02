import os
import re
import numpy as np
import pandas as pd


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between the true and predicted image intensities.

    Args:
        y_true (np.ndarray): The ground truth image data (normalized).
        y_pred (np.ndarray): The predicted image data (normalized).

    Returns:
        float: The calculated RMSE value.
    """
    # Ensure inputs are numpy arrays and flatten them for element-wise comparison
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    # Calculate Mean Squared Error
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)

    # Return Root Mean Squared Error
    return np.sqrt(mse)


def save_submission_file(predictions, output_path):
    """
    Formats predictions into the required submission format and saves to a CSV file.

    The format requires melting each image into pixels with an ID of 'image_row_col'.
    Rows and columns are 1-based.

    Args:
        predictions (dict): A dictionary where keys are image filenames (e.g., '110.png')
                            and values are numpy arrays of the denoised image (H, W).
        output_path (str): The full path where the submission CSV file should be saved.
    """
    submission_rows = []

    # Process images in sorted order of filenames for deterministic output
    for filename in sorted(predictions.keys()):
        image = predictions[filename]

        # Extract numeric image ID from filename (e.g., '110.png' -> '110')
        match = re.search(r"(\d+)", filename)
        if match:
            image_id = match.group(1)
        else:
            # Fallback if no digits found
            image_id = os.path.splitext(filename)[0]

        # Ensure image is 2D (remove channel dimension if present)
        if image.ndim == 3:
            image = image.squeeze()

        # Clip pixel values to valid range [0, 1]
        image = np.clip(image, 0, 1)

        height, width = image.shape

        # Create 1-based coordinate grids
        # np.indices returns indices for grid (0-based), so we add 1
        row_indices, col_indices = np.indices((height, width))
        row_indices += 1
        col_indices += 1

        # Flatten arrays for DataFrame construction
        flat_values = image.flatten()
        flat_rows = row_indices.flatten()
        flat_cols = col_indices.flatten()

        # Create a temporary DataFrame for this image
        df_img = pd.DataFrame({"r": flat_rows, "c": flat_cols, "value": flat_values})

        # Construct the 'id' column: image_row_col
        # Using vectorized string concatenation for efficiency
        df_img["id"] = (
            str(image_id)
            + "_"
            + df_img["r"].astype(str)
            + "_"
            + df_img["c"].astype(str)
        )

        # Keep only required columns
        submission_rows.append(df_img[["id", "value"]])

    # Concatenate all image DataFrames
    if submission_rows:
        full_submission = pd.concat(submission_rows, ignore_index=True)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        full_submission.to_csv(output_path, index=False)
    else:
        # Handle empty predictions case
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("id,value\n")
