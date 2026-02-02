import os
import random
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_f1(y_true, y_pred, average="samples"):
    """
    Calculates the Mean F1-Score (Samples F1) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N_samples, N_classes).
        y_pred (np.ndarray): Predicted binary labels (N_samples, N_classes).
        average (str): The averaging strategy. Defaults to 'samples' for Mean F1.

    Returns:
        float: The calculated F1 score.
    """
    # zero_division=0 ensures no errors if a sample has no tags (though unlikely in this dataset)
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def save_checkpoint(model, optimizer, epoch, score, path=Config.MODEL_SAVE_PATH):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current training epoch.
        score (float): Validation score at this checkpoint.
        path (str): Destination path for the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "score": score,
    }
    torch.save(state, path)


def load_checkpoint(
    model, optimizer=None, path=Config.MODEL_SAVE_PATH, device=Config.DEVICE
):
    """
    Loads model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        path (str): Path to the checkpoint file.
        device (str): Device to map the storage to.

    Returns:
        dict: The full checkpoint dictionary (including epoch and score).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def load_or_create(
    file_path, compute_func, load_cached_data=True, file_type="npy", **kwargs
):
    """
    Generic caching mechanism for deterministic data processing.

    Logic:
    1. IF load_cached_data is True: Try to load the file.
    2. IF loading fails OR load_cached_data is False:
       - Compute data using compute_func(**kwargs).
       - Save result to file_path.
    3. Return data.

    Args:
        file_path (str): Path to the cache file.
        compute_func (callable): Function to compute data if cache is missed.
        load_cached_data (bool): Whether to attempt loading from cache.
        file_type (str): Format of the file ('npy' or 'json').
        **kwargs: Arguments passed to compute_func.

    Returns:
        The loaded or computed data.
    """
    # Ensure directory exists
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    data = None
    loaded = False

    if load_cached_data and os.path.exists(file_path):
        try:
            if file_type == "npy":
                # allow_pickle=True is required for object arrays (e.g. string arrays)
                # Standard numpy format is used.
                data = np.load(file_path, allow_pickle=True)
            elif file_type == "json":
                with open(file_path, "r") as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")
            loaded = True
        except Exception as e:
            # If loading fails (corrupt file, etc.), we proceed to re-compute
            print(f"Warning: Failed to load cache from {file_path}. Error: {e}")
            loaded = False

    if not loaded:
        data = compute_func(**kwargs)

        if file_type == "npy":
            np.save(file_path, data)
        elif file_type == "json":
            with open(file_path, "w") as f:
                json.dump(data, f)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

    return data
