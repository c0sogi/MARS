import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

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
    logging.info(f"Random seed set to {seed}")


def setup_logging(level=logging.INFO):
    """
    Configures the root logger to print to stdout with a standard format.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_cache_dir() -> str:
    """
    Returns the working directory for caching, ensuring it exists.

    Returns:
        str: Path to the cache directory.
    """
    cache_dir = "./working/idea_58/"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_metadata(split: str, debug: bool = False):
    """
    Loads the metadata CSV for the specified split from the ./metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test'.
        debug (bool): If True, returns a small subsample of the data for debugging.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Split must be one of {valid_splits}")

    path = f"./metadata/{split}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    if debug:
        logging.info(f"Debug mode enabled: Loading subsample of {split} set.")
        df = df.head(20).copy()

    logging.info(f"Loaded {split} metadata: {df.shape}")
    return df


def save_submission(ids, probs, class_names, filename="./submission/submission.csv"):
    """
    Saves the submission file in the required format.

    Args:
        ids (array-like): Image IDs.
        probs (array-like): Probability matrix (N_samples, N_classes).
        class_names (list): List of class names corresponding to columns of probs.
        filename (str): Output filename.
    """
    # Ensure probabilities are within [0, 1] as per task requirements
    probs = np.clip(probs, 0, 1)

    # Create DataFrame
    submission = pd.DataFrame(probs, columns=class_names)
    submission.insert(0, "id", ids)

    # Ensure output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    submission.to_csv(filename, index=False)
    logging.info(f"Submission saved to {filename} with shape {submission.shape}")
