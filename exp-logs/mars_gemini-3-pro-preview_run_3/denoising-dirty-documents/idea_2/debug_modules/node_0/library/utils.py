import os
import torch
import numpy as np
import pandas as pd
from library.config import Config, seed_everything


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error between true and predicted values.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The RMSE value.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        loss (float): Validation loss (or metric) at this checkpoint.
        path (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device): Device to map the location to.

    Returns:
        dict: The checkpoint dictionary containing epoch, loss, etc.
        None: If the path does not exist.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def generate_submission_file(predictions, output_path=Config.SUBMISSION_PATH):
    """
    Generates the submission CSV file from a dictionary of predictions.

    Args:
        predictions (dict): A dictionary where keys are image filenames (e.g., '110.png')
                            and values are numpy arrays of the predicted image (H, W) or (H, W, 1).
                            Values should be in range [0, 1].
        output_path (str): Path to save the submission CSV.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ids = []
    values = []

    # Sort keys to ensure deterministic order (though not strictly required by CSV)
    sorted_filenames = sorted(predictions.keys())

    for filename in sorted_filenames:
        img_array = predictions[filename]

        # Ensure 2D array
        if img_array.ndim == 3:
            img_array = img_array.squeeze(-1)

        h, w = img_array.shape

        # Get image ID stem (e.g., '110.png' -> '110')
        img_id_stem = os.path.splitext(filename)[0]

        # Create grid of indices (1-based)
        # rows: 1..h, cols: 1..w
        # We repeat row indices for each column, and tile column indices for each row
        # Flattening order: row by row (standard C-order)

        # Generate row indices (1, 1, ..., 2, 2, ...)
        row_indices = np.repeat(np.arange(1, h + 1), w)

        # Generate col indices (1, 2, ..., 1, 2, ...)
        col_indices = np.tile(np.arange(1, w + 1), h)

        # Flatten pixel values
        flat_values = img_array.flatten()

        # Construct ID strings efficiently
        # Vectorized string formatting is tricky in numpy, list comprehension is usually fast enough for this scale
        # Format: {img_id}_{row}_{col}
        current_ids = [
            f"{img_id_stem}_{r}_{c}" for r, c in zip(row_indices, col_indices)
        ]

        ids.extend(current_ids)
        values.extend(flat_values)

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "value": values})

    # Save to CSV
    df.to_csv(output_path, index=False)
