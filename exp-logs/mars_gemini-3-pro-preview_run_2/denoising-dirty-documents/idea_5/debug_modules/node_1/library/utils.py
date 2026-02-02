import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The calculated RMSE.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def save_submission(predictions, save_path):
    """
    Formats predictions into the required CSV format and saves it.
    Melts 2D images into a pixel-wise list with IDs formatted as 'image_row_col'.
    Uses 1-based indexing for rows and columns.

    Args:
        predictions (dict): Dictionary mapping image_id (str) to predicted image (np.ndarray).
                            Images should be 2D arrays (H, W).
        save_path (str): The file path where the submission CSV will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ids = []
    values = []

    # Process images in sorted order for deterministic output
    for img_id in sorted(predictions.keys()):
        img = predictions[img_id]

        # Handle potential tensor or 3D inputs
        if hasattr(img, "cpu"):
            img = img.detach().cpu().numpy()
        if img.ndim == 3:
            img = img.squeeze()

        h, w = img.shape

        # Generate 1-based indices
        # np.indices returns grids for row and column indices
        # grid[0] varies rows (downwards), grid[1] varies columns (rightwards)
        grid = np.indices((h, w))
        row_indices = grid[0] + 1
        col_indices = grid[1] + 1

        # Flatten arrays (Row-Major / C-style default)
        # This produces sequence: (1,1), (1,2), (1,3)... which matches sampleSubmission.csv
        flat_rows = row_indices.flatten()
        flat_cols = col_indices.flatten()
        flat_vals = img.flatten()

        # Generate ID strings: "{image_id}_{row}_{col}"
        img_ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

        ids.extend(img_ids)
        values.extend(flat_vals)

    # Create DataFrame and save
    df = pd.DataFrame({"id": ids, "value": values})

    df.to_csv(save_path, index=False)
