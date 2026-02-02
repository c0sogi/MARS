import os
import pandas as pd
import numpy as np
import torch
from library.configuration import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    Delegates to the Config class implementation.

    Args:
        seed (int): The seed value to use.
    """
    Config.set_seed(seed)


def jaccard(str1, str2):
    """
    Calculates the Word-level Jaccard score between two strings.

    Formula: Intersection over Union of the set of words.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard score between 0.0 and 1.0.
    """
    # Ensure inputs are strings
    str1 = str(str1) if str1 is not None else ""
    str2 = str(str2) if str2 is not None else ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)

    if union_len == 0:
        return 0.0

    return float(len(c)) / union_len


def compute_average_jaccard(y_true, y_pred):
    """
    Computes the average Jaccard score for lists of ground truths and predictions.

    Args:
        y_true (list): List of ground truth strings.
        y_pred (list): List of predicted strings.

    Returns:
        float: The average Jaccard score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    scores = [jaccard(gt, pred) for gt, pred in zip(y_true, y_pred)]
    return np.mean(scores)


def clean_text(text):
    """
    Basic text cleaning utility.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return text.strip()


def load_data(split="train", nrows=None):
    """
    Loads the dataset metadata from the paths defined in Config.

    Args:
        split (str): One of 'train', 'val', 'test'.
        nrows (int, optional): Number of rows to load (for debugging/testing).

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if split == "train":
        path = Config.TRAIN_META_PATH
    elif split == "val":
        path = Config.VAL_META_PATH
    elif split == "test":
        path = Config.TEST_META_PATH
    else:
        raise ValueError("Split must be one of 'train', 'val', 'test'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path, nrows=nrows)

    # Ensure text columns are strings to avoid issues with NaN in text fields
    text_cols = ["context", "question"]
    if "answer_text" in df.columns:
        text_cols.append("answer_text")

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df
