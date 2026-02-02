import os
import random
import json
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score.
    """
    if str1 is None:
        str1 = ""
    if str2 is None:
        str2 = ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def compute_jaccard_score(ground_truths, predictions):
    """
    Computes the average Jaccard score for a list of ground truths and predictions.

    Args:
        ground_truths (list of str): List of expected answer strings.
        predictions (list of str): List of predicted answer strings.

    Returns:
        float: The average Jaccard score.
    """
    if not ground_truths or not predictions:
        return 0.0

    if len(ground_truths) != len(predictions):
        raise ValueError("Length of ground_truths and predictions must match.")

    scores = [jaccard(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)


def ensure_dir(path):
    """
    Ensures that the directory for the given path exists.
    If path is a file path, ensures the parent directory exists.
    If path is a directory path, ensures it exists.
    """
    # If it looks like a file (has an extension), get the parent dir
    if os.path.splitext(path)[1]:
        dirname = os.path.dirname(path)
    else:
        dirname = path

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def save_json(data, filepath):
    """
    Saves a dictionary or list to a JSON file.

    Args:
        data: The data to save.
        filepath (str): The path to the output file.
    """
    ensure_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filepath):
    """
    Loads data from a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        The loaded data (dict or list).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
