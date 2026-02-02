import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import set_seed, WORKING_DIR

# Import set_seed to expose it via utils as well, though it resides in config
__all__ = [
    "set_seed",
    "AverageMeter",
    "count_parameters",
    "calculate_accuracy",
    "save_checkpoint",
    "get_cached_numpy",
    "get_cached_df",
]


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg}"


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(output, target):
    """
    Computes the accuracy for multiclass classification.

    Args:
        output (torch.Tensor): Logits or probabilities of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size).

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(1.0 / batch_size).item()


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", best_filename="model_best.pth"
):
    """
    Saves a training checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Filename for the checkpoint.
        best_filename (str): Filename for the best model copy.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    filepath = os.path.join(WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(WORKING_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def get_cached_numpy(filename, compute_func, load_cached_data=True, *args, **kwargs):
    """
    Retrieves a numpy array from cache or computes it if missing/requested.

    Args:
        filename (str): Name of the file (e.g., 'features.npy').
        compute_func (callable): Function to compute the data if not cached.
        load_cached_data (bool): If True, attempts to load from disk first.
        *args, **kwargs: Arguments passed to compute_func.

    Returns:
        np.ndarray: The requested data.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    filepath = os.path.join(WORKING_DIR, filename)

    if load_cached_data and os.path.exists(filepath):
        try:
            print(f"Loading cached numpy array from {filepath}...")
            data = np.load(filepath)
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Computing data for {filename}...")
    data = compute_func(*args, **kwargs)

    if not isinstance(data, np.ndarray):
        raise ValueError("compute_func must return a numpy array.")

    np.save(filepath, data)
    print(f"Saved computed data to {filepath}")

    return data


def get_cached_df(filename, compute_func, load_cached_data=True, *args, **kwargs):
    """
    Retrieves a pandas DataFrame from cache (parquet) or computes it if missing/requested.

    Args:
        filename (str): Name of the file (e.g., 'metadata.parquet').
        compute_func (callable): Function to compute the dataframe if not cached.
        load_cached_data (bool): If True, attempts to load from disk first.
        *args, **kwargs: Arguments passed to compute_func.

    Returns:
        pd.DataFrame: The requested dataframe.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    filepath = os.path.join(WORKING_DIR, filename)

    # Ensure filename ends with .parquet for consistency
    if not filepath.endswith(".parquet"):
        filepath += ".parquet"

    if load_cached_data and os.path.exists(filepath):
        try:
            print(f"Loading cached dataframe from {filepath}...")
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Computing dataframe for {filename}...")
    df = compute_func(*args, **kwargs)

    if not isinstance(df, pd.DataFrame):
        raise ValueError("compute_func must return a pandas DataFrame.")

    df.to_parquet(filepath, index=False)
    print(f"Saved computed dataframe to {filepath}")

    return df
