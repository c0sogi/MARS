import os
import torch
import numpy as np
import random
from library.config import Config, seed_everything


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the provided configuration function.
    """
    seed_everything(seed)


def calculate_rmse(predictions, targets):
    """
    Computes the Root Mean Squared Error (RMSE) between predicted and actual pixel intensities.

    Args:
        predictions (torch.Tensor or np.ndarray): The predicted pixel intensities.
        targets (torch.Tensor or np.ndarray): The ground truth pixel intensities.

    Returns:
        float: The RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Flatten arrays to ensure element-wise operation regardless of shape dimensions
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()

    # Calculate MSE then RMSE
    mse = np.mean((pred_flat - targ_flat) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def save_checkpoint(model, optimizer, epoch, loss, filename=Config.MODEL_SAVE_PATH):
    """
    Saves the model training checkpoint including model state, optimizer state, epoch, and loss.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current training epoch.
        loss (float): The current validation loss.
        filename (str): Path to save the checkpoint file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filename)


def normalize_image(image):
    """
    Normalizes image pixel intensities from [0, 255] to [0, 1].

    Args:
        image (np.ndarray): Input image array (uint8 or similar).

    Returns:
        np.ndarray: Normalized image array (float32).
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image):
    """
    Denormalizes image pixel intensities from [0, 1] back to [0, 255].
    Used for saving images to disk or visualization.

    Args:
        image (np.ndarray): Input normalized image array.

    Returns:
        np.ndarray: Denormalized image array (uint8).
    """
    # Clip values to ensure they fall within valid range before scaling
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def get_cached_data(
    cache_filename,
    compute_func,
    load_cached_data=True,
    cache_dir=Config.WORKING_DIR,
    **kwargs
):
    """
    Retrieves data from a local cache file if it exists; otherwise, computes the data
    using the provided function and saves it to the cache.

    Strictly uses .npy format for caching (no pickle).

    Args:
        cache_filename (str): Name of the cache file (e.g., 'data.npy').
        compute_func (callable): Function to compute the data if cache is missing.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_dir (str): Directory where cache files are stored.
        **kwargs: Arguments to pass to compute_func.

    Returns:
        np.ndarray: The loaded or computed data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Try to load the file if requested
    if load_cached_data and os.path.exists(cache_path):
        try:
            # allow_pickle=False ensures we strictly use npy format for security/compatibility
            return np.load(cache_path, allow_pickle=False)
        except Exception:
            # If loading fails (corrupt file), proceed to recompute
            pass

    # 2. Compute data from scratch
    data = compute_func(**kwargs)

    # 3. Save result to cache
    np.save(cache_path, data)

    return data
