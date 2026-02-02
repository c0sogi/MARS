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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth images or pixels.
        y_pred (np.ndarray or torch.Tensor): Predicted images or pixels.

    Returns:
        float: The calculated RMSE value.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure data is float for accurate calculation
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def save_submission(predictions, submission_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the required CSV format and saves them.

    The format requires melting images into pixels with ids: {image_id}_{row}_{col}.
    Indices are 1-based (row 1, col 1 is top-left).

    Args:
        predictions (dict): A dictionary where keys are image filenames (e.g., '110.png')
                            and values are numpy arrays of the denoised image (H, W).
                            Values should be in the range [0, 1].
        submission_path (str): The file path to save the submission CSV.
    """
    ids = []
    values = []

    # Sort keys to ensure deterministic order of images in the file
    sorted_filenames = sorted(predictions.keys())

    for filename in sorted_filenames:
        img = predictions[filename]

        # Handle potential channel dimension (H, W, 1) -> (H, W)
        if img.ndim == 3:
            img = img.squeeze()

        # Clip values to valid range [0, 1] as per requirements
        img = np.clip(img, 0, 1)

        h, w = img.shape

        # Generate 1-based indices
        # np.indices returns indices in (row_indices, col_indices) order
        grid_indices = np.indices((h, w))

        # Flatten everything in row-major order (default for flatten)
        # Add 1 to convert 0-based index to 1-based index
        row_indices = grid_indices[0].flatten() + 1
        col_indices = grid_indices[1].flatten() + 1
        pixel_values = img.flatten()

        # Extract image ID (remove extension, e.g., '110.png' -> '110')
        image_id_str = os.path.splitext(filename)[0]

        # Generate ID strings: {image_id}_{row}_{col}
        # Using list comprehension is efficient enough for the dataset size
        current_ids = [
            f"{image_id_str}_{r}_{c}" for r, c in zip(row_indices, col_indices)
        ]

        ids.extend(current_ids)
        values.extend(pixel_values)

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "value": values})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Save to CSV
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
