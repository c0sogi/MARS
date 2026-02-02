import os
import random
import numpy as np
import torch
import logging
import sys
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler_list = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handler_list.append(file_handler)

    for handler in handler_list:
        handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    for handler in handler_list:
        logger.addHandler(handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

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


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model checkpoint.
    """

    def __init__(
        self,
        patience=3,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # Printing full precision as requested
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def load_or_create_cache(file_path, compute_func, load_cached_data=True, **kwargs):
    """
    Generic caching mechanism.

    Args:
        file_path (str): Path to the cache file (must end in .parquet or .npy).
        compute_func (callable): Function to compute data if cache is missing.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to compute_func.

    Returns:
        The loaded or computed data.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 1. Try to load
    if load_cached_data and os.path.exists(file_path):
        print(f"Loading cached data from {file_path}...")
        try:
            if file_path.endswith(".parquet"):
                return pd.read_parquet(file_path)
            elif file_path.endswith(".npy"):
                return np.load(file_path, allow_pickle=True)
            else:
                # Fallback or error, but we stick to requested formats
                raise ValueError(
                    "Unsupported cache file extension. Use .parquet or .npy"
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute
    print(f"Computing data (Cache miss or force recompute)...")
    data = compute_func(**kwargs)

    # 3. Save
    print(f"Saving data to cache at {file_path}...")
    if file_path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, index=False)
        else:
            raise TypeError("Data must be a DataFrame for .parquet cache.")
    elif file_path.endswith(".npy"):
        np.save(file_path, data)
    else:
        raise ValueError("Unsupported cache file extension. Use .parquet or .npy")

    return data


def save_submission(ids, predictions, output_path):
    """
    Saves the submission file in the required format.

    Args:
        ids (list): List of id strings (e.g., '0_0').
        predictions (list): List of predicted strings.
        output_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "after": predictions})

    # Ensure 'after' column is treated as string to avoid issues with quoting
    df["after"] = df["after"].astype(str)

    # Save to CSV
    # The sample submission uses double quotes for the 'after' column.
    # Pandas to_csv handles this automatically for strings containing special chars,
    # but we can verify against the sample format.
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def calculate_accuracy(preds, targets):
    """
    Calculates exact match accuracy.
    """
    if len(preds) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    correct = sum(p == t for p, t in zip(preds, targets))
    return correct / len(targets)
