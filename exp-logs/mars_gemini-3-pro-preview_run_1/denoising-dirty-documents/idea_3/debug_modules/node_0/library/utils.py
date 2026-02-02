import os
import cv2
import torch
import numpy as np
import pandas as pd
from library.config import Config, seed_everything


def get_metadata(split):
    """
    Loads the metadata DataFrame for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def load_image(path):
    """
    Loads an image from a given path, converts it to grayscale,
    and normalizes pixel intensities to [0, 1].

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Normalized grayscale image of shape (H, W).
    """
    # Load as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    return img


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Supports both numpy arrays and torch tensors.

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


def print_metric(name, value):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{name}: {value:.20f}")


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Saves the model checkpoint including optimizer state and training metadata.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch.
        loss (float): Current validation loss.
        filename (str): Name of the file to save (e.g., 'best_model.pth').
    """
    # Ensure directory exists
    save_dir = Config.WORKING_DIR
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, filename)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(state, file_path)


def create_submission_csv(predictions, output_path):
    """
    Generates the submission CSV file from a dictionary of predictions.

    Format:
    id,value
    110_1_1,1
    ...

    Args:
        predictions (dict): Dictionary where keys are image IDs (str)
                            and values are predicted image arrays (np.ndarray).
        output_path (str): Path to save the CSV file.
    """
    ids_list = []
    values_list = []

    # Sort keys to ensure deterministic order if needed, though not strictly required
    sorted_ids = sorted(predictions.keys())

    for img_id in sorted_ids:
        img = predictions[img_id]

        # Ensure image is 2D
        if len(img.shape) == 3:
            img = img.squeeze()

        h, w = img.shape

        # Create 1-based indices for rows and columns
        # We want row indices to repeat for each column (1,1,1..., 2,2,2...)
        # We want col indices to tile for each row (1,2,3..., 1,2,3...)
        # This corresponds to row-major flattening (C-style), which is default in numpy
        rows = np.repeat(np.arange(1, h + 1), w)
        cols = np.tile(np.arange(1, w + 1), h)

        # Flatten the image values
        flat_vals = img.flatten()

        # Generate ID strings: "{img_id}_{row}_{col}"
        # Using list comprehension is generally fast enough for this scale
        img_ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

        ids_list.extend(img_ids)
        values_list.extend(flat_vals)

    # Create DataFrame
    df = pd.DataFrame({"id": ids_list, "value": values_list})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
