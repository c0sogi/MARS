import os
import sys
import random
import logging
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name="project"):
    """
    Configures and returns a logger instance that writes to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Prevent propagation to root logger to avoid double logging
        logger.propagate = False

    return logger


def get_fine_grained_labels():
    """
    Scans the training directory to identify all unique fine-grained class labels.
    Excludes '_background_noise_' and ensures 'silence' is included.

    Returns:
        list: Sorted list of string labels (e.g., ['bed', 'bird', ..., 'yes', 'zero']).
    """
    train_dir = Config.TRAIN_AUDIO_DIR

    # Safety check if directory exists
    if not os.path.exists(train_dir):
        return []

    # Get all subdirectories (labels)
    subdirs = [
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    ]

    labels = []
    for d in subdirs:
        # _background_noise_ is handled separately as 'silence'
        if d == "_background_noise_":
            continue
        labels.append(d)

    # Ensure 'silence' is present (derived from background noise or empty clips)
    if Config.SILENCE_LABEL not in labels:
        labels.append(Config.SILENCE_LABEL)

    return sorted(labels)


def get_label_map():
    """
    Creates a mapping from fine-grained training labels to the 12 competition target labels.

    Logic:
    - Target labels (yes, no, up, down...) -> Keep as is.
    - 'silence' -> 'silence'.
    - All other labels (bed, bird, etc.) -> 'unknown'.

    Returns:
        dict: A dictionary where key is the fine-grained label and value is the target label.
    """
    fine_labels = get_fine_grained_labels()
    mapping = {}

    target_set = set(Config.TARGET_LABELS)

    for label in fine_labels:
        if label in target_set:
            mapping[label] = label
        elif label == Config.SILENCE_LABEL:
            mapping[label] = Config.SILENCE_LABEL
        else:
            mapping[label] = Config.UNKNOWN_LABEL

    return mapping


def save_submission(predictions, filenames, output_path=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        predictions (list or np.array): List of predicted label strings.
        filenames (list): List of corresponding filenames (e.g., 'clip_000.wav').
        output_path (str): Full path to save the CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({"fname": filenames, "label": predictions})

    # Ensure correct column order
    df = df[["fname", "label"]]

    # Save to CSV
    df.to_csv(output_path, index=False)
