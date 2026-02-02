import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for the ensemble strategy.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted pixel intensities.

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

    # Ensure inputs are flattened or same shape for element-wise operation
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the specified file.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def load_or_process_cache(cache_filename, process_fn, load_cached_data=True, **kwargs):
    """
    Generic caching mechanism for deterministic data processing.
    Uses .npz (numpy) for storage to avoid pickle security/compatibility issues.

    Args:
        cache_filename (str): Name of the cache file (e.g., 'train_cache.npz').
        process_fn (callable): Function to compute data if cache is missing.
                               Must return a dictionary of numpy arrays.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments to pass to process_fn.

    Returns:
        dict: The loaded or processed data (dictionary of arrays).
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            # allow_pickle=False ensures we strictly use NPY format
            with np.load(cache_path, allow_pickle=False) as loaded:
                # Convert NpzFile to standard dict to keep data in memory
                return {key: loaded[key] for key in loaded.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data (Cache miss or force reload)...")
    data = process_fn(**kwargs)

    # Validation: Ensure data is a dict of arrays for np.savez
    if not isinstance(data, dict):
        raise ValueError(
            "process_fn must return a dictionary of numpy arrays for caching."
        )

    print(f"Saving data to {cache_path}...")
    np.savez_compressed(cache_path, **data)

    return data


def create_submission_file(predictions_dict, output_path=Config.SUBMISSION_FILE):
    """
    Formats predictions into the required submission CSV format.
    Melts images into pixels with id 'image_row_col'.

    Args:
        predictions_dict (dict): Dictionary mapping image_id (str) to predicted image (np.ndarray).
                                 Image array should be shape (H, W) or (H, W, 1).
        output_path (str): Path to save the submission CSV.
    """
    records = []

    print("Formatting submission data...")
    # Sort keys for deterministic order
    for img_id in sorted(predictions_dict.keys()):
        img = predictions_dict[img_id]

        # Ensure image is 2D
        if img.ndim == 3:
            img = img.squeeze()

        rows, cols = img.shape

        # Vectorized ID generation
        # Create grid of indices (0-based)
        r_indices, c_indices = np.indices((rows, cols))

        # Convert to 1-based indexing for submission format (image_row_col)
        r_indices = r_indices + 1
        c_indices = c_indices + 1

        # Flatten arrays
        r_flat = r_indices.flatten()
        c_flat = c_indices.flatten()
        v_flat = img.flatten()

        # Generate ID strings: "{img_id}_{row}_{col}"
        # Using list comprehension is efficient enough for this scale
        ids = [f"{img_id}_{r}_{c}" for r, c in zip(r_flat, c_flat)]

        records.extend(zip(ids, v_flat))

    print(f"Saving submission to {output_path}...")
    # Create DataFrame and save
    df = pd.DataFrame(records, columns=["id", "value"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved. Shape: {df.shape}")
