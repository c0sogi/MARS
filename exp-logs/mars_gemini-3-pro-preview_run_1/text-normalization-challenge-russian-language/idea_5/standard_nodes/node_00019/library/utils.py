import os
import random
import re
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def should_route_to_neural(text: str) -> bool:
    """
    Determines if a token should be routed to the neural model.
    Includes tokens with digits (e.g. "123", "3rd") or Latin characters (e.g. "TV", "DVD")
    which often require generative normalization or transliteration.
    Cite solution_lesson_node_00010

    Args:
        text (str): The input token text.

    Returns:
        bool: True if the text contains digits or Latin letters, False otherwise.
    """
    if not isinstance(text, str):
        return False
    # Regex: \d for digits, [a-zA-Z] for Latin letters
    return bool(re.search(r"[\d]|[a-zA-Z]", text))


def get_device() -> torch.device:
    """
    Utility to get the available device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_exact_match_accuracy(preds: list, targets: list) -> float:
    """
    Calculates the accuracy metric: percentage of tokens where prediction matches target exactly.

    Args:
        preds (list): List of predicted strings.
        targets (list): List of ground truth strings.

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    if len(preds) != len(targets):
        # Fallback for mismatched lengths, though pipeline should prevent this
        return 0.0

    if len(preds) == 0:
        return 0.0

    correct_count = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct_count / len(preds)
