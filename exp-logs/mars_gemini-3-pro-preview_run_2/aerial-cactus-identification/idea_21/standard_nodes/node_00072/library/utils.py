import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(ids, probabilities, save_path=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.array): List of image IDs (filenames).
        probabilities (list or np.array): List of predicted probabilities for 'has_cactus'.
        save_path (str): Path to save the CSV file. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create the DataFrame
    submission_df = pd.DataFrame({"id": ids, "has_cactus": probabilities})

    # Save to CSV without index
    submission_df.to_csv(save_path, index=False)


def load_metadata(split):
    """
    Loads the metadata DataFrame for a specific split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_metrics(metrics):
    """
    Prints metric values with full precision.

    Args:
        metrics (dict): A dictionary of metric names and values.
    """
    for k, v in metrics.items():
        print(f"{k}: {v}")
