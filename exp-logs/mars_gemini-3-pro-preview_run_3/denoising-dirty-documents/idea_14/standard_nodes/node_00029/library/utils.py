import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth pixel intensities.
        y_pred (np.ndarray): Predicted pixel intensities.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def create_submission_file(predictions, output_path=Config.SUBMISSION_FILE):
    """
    Generates a submission CSV file from the predictions.

    Args:
        predictions (dict): A dictionary where keys are image IDs (e.g., '110.png' or '110')
                            and values are numpy arrays of the predicted denoised images.
        output_path (str): Path to save the submission CSV file.
    """
    data_frames = []

    # Sort keys for deterministic order
    sorted_keys = sorted(predictions.keys())

    for img_name in sorted_keys:
        # Extract ID (remove extension if present)
        # e.g., '110.png' -> '110'
        img_id = os.path.splitext(img_name)[0]

        img_data = predictions[img_name]

        # Ensure pixel values are in [0, 1]
        img_data = np.clip(img_data, 0, 1)

        h, w = img_data.shape

        # Create coordinate grids (1-based indexing as per task description)
        # indexing='ij' ensures:
        # r_grid varies along rows (1, 2, 3...)
        # c_grid varies along columns (1, 2, 3...)
        r_grid, c_grid = np.meshgrid(
            np.arange(1, h + 1), np.arange(1, w + 1), indexing="ij"
        )

        # Flatten arrays to create columns
        r_flat = r_grid.flatten()
        c_flat = c_grid.flatten()
        val_flat = img_data.flatten()

        # Create a temporary DataFrame for efficient string operation
        df_img = pd.DataFrame({"r": r_flat, "c": c_flat, "value": val_flat})

        # Vectorized string concatenation for ID
        # Format: {img_id}_{row}_{col}
        df_img["id"] = (
            img_id + "_" + df_img["r"].astype(str) + "_" + df_img["c"].astype(str)
        )

        # Keep only required columns
        data_frames.append(df_img[["id", "value"]])

    # Concatenate all image dataframes
    if data_frames:
        final_df = pd.concat(data_frames, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=["id", "value"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    final_df.to_csv(output_path, index=False)
