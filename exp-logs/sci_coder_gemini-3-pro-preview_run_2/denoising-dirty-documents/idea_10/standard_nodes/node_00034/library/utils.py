import os
import random
import numpy as np
import torch
import pandas as pd
import cv2
from sklearn.metrics import mean_squared_error


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (cuda or cpu).

    Returns:
        torch.device: The device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))


def save_checkpoint(state: dict, filename: str):
    """
    Saves the model training checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The path to save the checkpoint to.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None
) -> dict:
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    device = get_device()
    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")

    path = os.path.join("./metadata", f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Reads an image, converts to grayscale, and normalizes to [0, 1].

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The processed image as a float32 numpy array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Handle channels
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0
    return img


def load_image_with_cache(
    image_path: str, cache_path: str, load_cached_data: bool = True
) -> np.ndarray:
    """
    Loads an image with caching mechanism.

    Args:
        image_path (str): Path to the raw image file.
        cache_path (str): Path where the processed .npy file should be stored/loaded from.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: The processed image data.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            # If load fails, fall through to re-processing
            pass

    # 2. Process from scratch
    img_data = preprocess_image(image_path)

    # 3. Save to cache
    np.save(cache_path, img_data)

    return img_data


def save_submission(predictions: dict, output_path: str):
    """
    Formats and saves the submission file according to the pixel melting requirement.

    Args:
        predictions (dict): Dictionary mapping image_id (str) to predicted numpy array (H, W).
                            Values should be in range [0, 1].
        output_path (str): Path to save the submission CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data_list = []

    # Process each image
    # Sort keys for deterministic output order
    for img_id in sorted(predictions.keys()):
        img_arr = predictions[img_id]

        # Ensure 2D and clip values
        if len(img_arr.shape) == 3:
            img_arr = img_arr.squeeze()
        img_arr = np.clip(img_arr, 0, 1)

        rows, cols = img_arr.shape

        # Create coordinate grids (1-based indexing)
        r_indices, c_indices = np.indices((rows, cols))
        r_indices += 1
        c_indices += 1

        # Flatten arrays
        flat_vals = img_arr.flatten()
        flat_rows = r_indices.flatten()
        flat_cols = c_indices.flatten()

        # Create IDs: {img_id}_{row}_{col}
        ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

        # Create DataFrame for this image
        df_img = pd.DataFrame({"id": ids, "value": flat_vals})
        data_list.append(df_img)

    # Concatenate all and save
    if data_list:
        final_df = pd.concat(data_list, ignore_index=True)
        final_df.to_csv(output_path, index=False)
    else:
        # Handle empty case just in case
        with open(output_path, "w") as f:
            f.write("id,value\n")
