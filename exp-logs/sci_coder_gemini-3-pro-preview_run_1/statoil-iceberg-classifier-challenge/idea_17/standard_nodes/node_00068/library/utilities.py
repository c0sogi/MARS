import os
import random
import shutil
import numpy as np
import torch
import pandas as pd
from library.configuration import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, epoch, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, optimizer=None, filename="best_model.pth"):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): The filename of the checkpoint to load.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    if not os.path.exists(filepath):
        # Try looking in working dir directly if not in checkpoint dir
        filepath_alt = os.path.join(Config.WORKING_DIR, filename)
        if os.path.exists(filepath_alt):
            filepath = filepath_alt
        else:
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    # Handle state dict loading (handling potential DataParallel wrapping)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # Assume it is the state dict directly

    # Remove 'module.' prefix if it exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """Logs training metrics to console and file."""

    def __init__(self, filename="training_log.txt"):
        self.log_path = os.path.join(Config.WORKING_DIR, filename)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        # Initialize file
        with open(self.log_path, "w") as f:
            f.write("Training Log\n")

    def log(self, message):
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def log_metrics(self, epoch, metrics, phase="Train"):
        """
        Logs metrics with full precision.
        metrics: dict of metric_name -> value
        """
        msg_parts = [f"Epoch: {epoch}", f"Phase: {phase}"]
        for k, v in metrics.items():
            msg_parts.append(f"{k}: {v}")

        message = " | ".join(msg_parts)
        self.log(message)


def get_or_create_cached_array(filename, compute_func, load_cached_data=True, **kwargs):
    """
    Retrieves a numpy array from cache or computes it if missing/forced.

    Args:
        filename (str): Name of the cache file (e.g., 'data.npy').
        compute_func (callable): Function to compute data if cache miss.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to compute_func.

    Returns:
        np.ndarray: The requested data.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, filename)

    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to recompute if load fails

    # Compute data
    data = compute_func(**kwargs)

    # Save to cache
    np.save(cache_path, data)

    return data


def save_submission(ids, predictions, filename="submission.csv"):
    """
    Saves predictions to a CSV file in the submission format.

    Args:
        ids (list or np.array): List of image IDs.
        predictions (list or np.array): List of probabilities (is_iceberg).
        filename (str): Output filename.
    """
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    df = pd.DataFrame({"id": ids, "is_iceberg": predictions})

    # Ensure correct column order and format
    output_path = os.path.join(Config.SUBMISSION_DIR, filename)
    df.to_csv(output_path, index=False)
    # print(f"Submission saved to {output_path}")
