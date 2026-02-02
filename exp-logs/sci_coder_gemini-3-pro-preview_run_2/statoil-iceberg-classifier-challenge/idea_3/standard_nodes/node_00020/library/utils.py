import os
import torch
import numpy as np
import pandas as pd
from library.config import Config, set_seed


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.

    Args:
        model (torch.nn.Module): The model instance to save.
        path (str): The file path where the checkpoint will be saved.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device=Config.DEVICE):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path of the checkpoint.
        device (str): The device to map the location to (e.g., 'cpu', 'cuda').

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def save_submission(ids, predictions, path):
    """
    Saves the predictions to a CSV file in the required submission format.

    Args:
        ids (list or np.array): List of image IDs.
        predictions (list or np.array): List of predicted probabilities.
        path (str): The file path for the submission CSV.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame({"id": ids, "is_iceberg": predictions})
    df.to_csv(path, index=False)


def load_or_process_cache(cache_path, process_fn, load_cache=True, **kwargs):
    """
    Generic caching mechanism for numpy data.

    Args:
        cache_path (str): Path to the .npz cache file.
        process_fn (callable): Function to execute if cache is missing or load_cache is False.
                               Must return a dictionary of numpy arrays.
        load_cache (bool): If True, attempts to load from cache first.
        **kwargs: Keyword arguments passed to process_fn.

    Returns:
        dict: A dictionary containing the numpy arrays.
    """
    directory = os.path.dirname(cache_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if load_cache and os.path.exists(cache_path):
        try:
            # Load using numpy
            data = np.load(cache_path)
            # Convert NpzFile to dict to ensure data is fully loaded into memory
            return dict(data)
        except Exception as e:
            # If loading fails (e.g. corruption), proceed to re-process
            pass

    # Process data from scratch
    data_dict = process_fn(**kwargs)

    # Save to cache using compression
    np.savez_compressed(cache_path, **data_dict)

    return data_dict


def print_metrics(metrics, prefix=""):
    """
    Prints a dictionary of metrics with full precision.

    Args:
        metrics (dict): Dictionary where keys are metric names and values are scores.
        prefix (str): Optional string to prepend to the output.
    """
    metrics_str = " ".join([f"{k}: {v}" for k, v in metrics.items()])
    if prefix:
        print(f"{prefix} {metrics_str}")
    else:
        print(metrics_str)
