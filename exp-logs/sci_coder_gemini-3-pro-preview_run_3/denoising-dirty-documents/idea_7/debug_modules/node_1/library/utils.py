import os
import random
import numpy as np
import torch
import cv2
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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_grayscale(path):
    """
    Loads an image from the specified path in grayscale, normalizes it to [0, 1],
    and returns it as a PyTorch tensor.

    Args:
        path (str): Path to the image file.

    Returns:
        torch.Tensor: Tensor of shape (1, H, W) with values in [0, 1].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    # Load image in grayscale mode
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"Failed to load image at {path}. The file might be corrupted."
        )

    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert to tensor and add channel dimension: (H, W) -> (1, H, W)
    tensor = torch.from_numpy(img).unsqueeze(0)

    return tensor


def calculate_rmse(pred, target):
    """
    Calculates the Root Mean Squared Error (RMSE) between prediction and target tensors.

    Args:
        pred (torch.Tensor): Predicted pixel intensities.
        target (torch.Tensor): Ground truth pixel intensities.

    Returns:
        torch.Tensor: Scalar tensor containing the RMSE value.
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    return rmse


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current training epoch.
        loss (float): Current validation loss.
        path (str): Destination path for the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=Config.DEVICE):
    """
    Loads model and optimizer state from a checkpoint file.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        tuple: (epoch, loss) from the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss


def create_submission(predictions_dict, output_path):
    """
    Formats the predictions into the required CSV format for submission.
    The format melts the image into pixels with id 'image_row_col'.

    Args:
        predictions_dict (dict): Dictionary mapping image_id (str) to predicted image (numpy array or tensor).
                                 image_id should be the filename without extension (e.g., '110').
        output_path (str): Path to save the submission CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use a buffered writer for efficiency given the large number of lines
    with open(output_path, "w") as f:
        f.write("id,value\n")

        # Sort keys to ensure deterministic output order
        sorted_ids = sorted(predictions_dict.keys())

        for img_id in sorted_ids:
            img = predictions_dict[img_id]

            # Ensure image is a numpy array
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            # Remove channel dimension if present: (1, H, W) -> (H, W)
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)

            h, w = img.shape

            # Iterate through pixels
            # Note: Task specifies 1-based indexing for rows and columns
            for r in range(h):
                for c in range(w):
                    # Get pixel value and clamp to valid range [0, 1]
                    val = float(img[r, c])
                    val = max(0.0, min(1.0, val))

                    # Construct ID: image_row_col
                    row_idx = r + 1
                    col_idx = c + 1
                    pixel_id = f"{img_id}_{row_idx}_{col_idx}"

                    f.write(f"{pixel_id},{val}\n")
