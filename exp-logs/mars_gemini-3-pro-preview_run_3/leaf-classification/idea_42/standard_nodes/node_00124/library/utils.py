import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = "./working/idea_solution"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")


# -----------------------------------------------------------------------------
# Reproducibility & Environment
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (CUDA if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    """
    Ensures that a directory exists. If not, creates it.
    """
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(log_path: str = "execution.log", level=logging.INFO):
    """
    Configures logging to both console and a file.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# -----------------------------------------------------------------------------
# Data Loading & Path Management
# -----------------------------------------------------------------------------
def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for a specific split ('train', 'val', 'test').

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    path = os.path.join(METADATA_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)
    return df


def get_image_path(image_id: int) -> str:
    """
    Constructs the absolute path for a given image ID.
    """
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


# -----------------------------------------------------------------------------
# Caching Mechanism
# -----------------------------------------------------------------------------
def save_to_cache(data, filename: str, sub_dir: str = ""):
    """
    Saves data to the cache directory using .npy (numpy) or .parquet (pandas).

    Args:
        data: The object to save (numpy array or pandas DataFrame).
        filename (str): The name of the file.
        sub_dir (str): Optional subdirectory within the cache folder.
    """
    target_dir = os.path.join(CACHE_DIR, sub_dir)
    ensure_dir(target_dir)
    file_path = os.path.join(target_dir, filename)

    if isinstance(data, np.ndarray):
        np.save(file_path, data)
    elif isinstance(data, pd.DataFrame):
        data.to_parquet(file_path, index=False)
    else:
        raise TypeError("Data must be a numpy array or pandas DataFrame.")


def load_from_cache(filename: str, sub_dir: str = ""):
    """
    Loads data from the cache directory.

    Args:
        filename (str): The name of the file.
        sub_dir (str): Optional subdirectory within the cache folder.

    Returns:
        The loaded data, or None if the file does not exist.
    """
    target_dir = os.path.join(CACHE_DIR, sub_dir)
    file_path = os.path.join(target_dir, filename)

    if not os.path.exists(file_path):
        return None

    if filename.endswith(".npy"):
        return np.load(file_path, allow_pickle=True)
    elif filename.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        # Fallback check if extension was omitted in filename but present in logic
        if os.path.exists(file_path + ".npy"):
            return np.load(file_path + ".npy", allow_pickle=True)
        if os.path.exists(file_path + ".parquet"):
            return pd.read_parquet(file_path + ".parquet")
        return None


# -----------------------------------------------------------------------------
# Metrics & Submission
# -----------------------------------------------------------------------------
def calculate_metric(y_true, y_pred_probs):
    """
    Calculates the Multi-class Log Loss.

    Args:
        y_true: Array-like of ground truth labels (integers or strings).
        y_pred_probs: Array-like of predicted probabilities (shape: [n_samples, n_classes]).

    Returns:
        float: The log loss value.
    """
    # Ensure probabilities are clipped to avoid log(0)
    # The competition metric specifies: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)

    # Normalize rows to sum to 1 (as per metric description)
    row_sums = y_pred_probs.sum(axis=1)
    y_pred_probs = y_pred_probs / row_sums[:, np.newaxis]

    # Explicitly define labels to handle disjoint validation sets
    labels = np.arange(y_pred_probs.shape[1])
    return log_loss(y_true, y_pred_probs, labels=labels)


def save_submission(ids, classes, probs, filename="submission.csv"):
    """
    Generates and saves the submission file in the required format.

    Args:
        ids: Array-like of image IDs.
        classes: List of class names (column headers).
        probs: Array-like of predicted probabilities (shape: [n_samples, n_classes]).
        filename: Output filename.
    """
    # Ensure probabilities are clipped and normalized
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    row_sums = probs.sum(axis=1)
    probs = probs / row_sums[:, np.newaxis]

    # Create DataFrame
    df_sub = pd.DataFrame(probs, columns=classes)
    df_sub.insert(0, "id", ids)

    # Save
    output_dir = os.path.join(WORKING_DIR, "submission")
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, filename)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
