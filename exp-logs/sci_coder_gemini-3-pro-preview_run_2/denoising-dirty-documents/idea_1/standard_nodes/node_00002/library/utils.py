import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(predictions, targets):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted pixel intensities.
        targets (np.ndarray or torch.Tensor): Actual pixel intensities.

    Returns:
        float: The RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure data types are float32 for precision
    predictions = predictions.astype(np.float32)
    targets = targets.astype(np.float32)

    # Calculate MSE and then RMSE
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def save_submission_file(predictions, output_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the predictions to a CSV file in the required format.

    The format melts each image into a set of pixels with id 'image_row_col'.
    Rows and columns are 1-indexed.

    Args:
        predictions (dict): A dictionary where keys are image IDs (str) and
                          values are numpy arrays of the denoised images (H, W).
        output_path (str): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_ids = []
    all_values = []

    # Sort keys to ensure deterministic order of images in the file
    sorted_image_ids = sorted(predictions.keys())

    for img_id in sorted_image_ids:
        img_data = predictions[img_id]

        # Handle potential channel dimension (H, W, 1) -> (H, W)
        if img_data.ndim == 3:
            img_data = img_data.squeeze()

        h, w = img_data.shape

        # Create 1-based grid indices
        # rows: [[1, 1, ...], [2, 2, ...]]
        # cols: [[1, 2, ...], [1, 2, ...]]
        rows, cols = np.indices((h, w))
        rows = rows + 1
        cols = cols + 1

        # Flatten the arrays to create the list of pixels
        flat_rows = rows.flatten()
        flat_cols = cols.flatten()
        flat_values = img_data.flatten()

        # Generate IDs: "image_row_col"
        # Using list comprehension is generally efficient enough for this scale
        # and avoids complex string manipulation with numpy char arrays.
        img_ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

        all_ids.extend(img_ids)
        all_values.extend(flat_values)

    # Create a DataFrame
    df = pd.DataFrame({"id": all_ids, "value": all_values})

    # Save to CSV
    df.to_csv(output_path, index=False)
