import os
import random
import numpy as np
import torch
import pydicom
import cv2
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom(
    path,
    window_center=Config.WINDOW_CENTER,
    window_width=Config.WINDOW_WIDTH,
    output_size=None,
):
    """
    Loads a DICOM file, applies bone windowing, and normalizes to [0, 1].

    Args:
        path (str): Path to the DICOM file.
        window_center (int): Window center level.
        window_width (int): Window width.
        output_size (int, optional): If provided, resizes the image to (output_size, output_size).

    Returns:
        np.ndarray: 2D float32 array normalized to [0, 1].
    """
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)

        # Apply RescaleSlope and RescaleIntercept if they exist
        slope = getattr(dicom, "RescaleSlope", 1.0)
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Apply Windowing
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        if img_max != img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = img - img_min

        # Resize if requested
        if output_size is not None:
            if img.shape[0] != output_size or img.shape[1] != output_size:
                img = cv2.resize(
                    img, (output_size, output_size), interpolation=cv2.INTER_LINEAR
                )

        return img

    except Exception as e:
        # Return a black image of default size or requested size in case of error
        size = output_size if output_size else Config.ORIGINAL_SIZE
        return np.zeros((size, size), dtype=np.float32)


def get_bbox_from_mask(mask, margin=0):
    """
    Calculates the bounding box of a binary mask.

    Args:
        mask (np.ndarray): 2D binary mask.
        margin (int): Margin to add around the bounding box.

    Returns:
        list: [y_min, x_min, y_max, x_max] or None if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    h, w = mask.shape
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)

    return [y_min, x_min, y_max, x_max]


def get_soft_anatomical_map(mask, num_classes=7):
    """
    Converts a segmentation mask into a presence vector for C1-C7.

    Args:
        mask (np.ndarray): 2D mask with integer labels (1-7 for C1-C7).
        num_classes (int): Number of vertebrae classes.

    Returns:
        np.ndarray: Float array of shape (num_classes,) with 1.0 if class exists in mask.
    """
    presence = np.zeros(num_classes, dtype=np.float32)
    if mask is None:
        return presence

    unique_labels = np.unique(mask)
    for label in unique_labels:
        label = int(label)
        if 1 <= label <= num_classes:
            presence[label - 1] = 1.0
    return presence


def calculate_weighted_log_loss(y_true, y_pred, device=Config.DEVICE):
    """
    Calculates the weighted multi-label logarithmic loss.

    Args:
        y_true (torch.Tensor or np.ndarray): True labels, shape (N, 8).
                                             Order: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities, shape (N, 8).
        device (str): Device to perform calculation on.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32).to(device)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32).to(device)

    # Clip predictions to avoid log(0)
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    # Define weights corresponding to [C1, C2, C3, C4, C5, C6, C7, patient_overall]
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0], device=device)

    # Binary Cross Entropy
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    # Apply weights
    weighted_loss = loss * weights

    # Average across all rows (N samples * 8 targets)
    # The metric definition says "loss is averaged across all rows".
    # This implies mean() over the entire tensor.
    return weighted_loss.mean()


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint.
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
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path, optimizer=None, device=Config.DEVICE):
    """
    Loads the model checkpoint.
    """
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found.")
        return None

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {path} (Epoch {checkpoint.get('epoch', 'Unknown')})")
    return checkpoint


def save_to_cache(data, filename, use_parquet=False):
    """
    Saves data to the cache directory.
    """
    path = os.path.join(Config.CACHE_DIR, filename)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if use_parquet and isinstance(data, pd.DataFrame):
        data.to_parquet(path, index=False)
    else:
        np.save(path, data)
    print(f"Cached data saved to {path}")


def load_from_cache(filename, use_parquet=False):
    """
    Loads data from the cache directory if it exists.
    """
    path = os.path.join(Config.CACHE_DIR, filename)
    if not os.path.exists(path):
        return None

    if use_parquet:
        return pd.read_parquet(path)
    else:
        return np.load(path, allow_pickle=True)
