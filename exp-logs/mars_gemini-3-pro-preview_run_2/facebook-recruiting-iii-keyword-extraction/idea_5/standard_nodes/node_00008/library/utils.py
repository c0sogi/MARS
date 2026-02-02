import os
import sys
import time
import random
import re
import contextlib
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_text(text):
    """
    Cleans the input text by stripping HTML tags and converting to lowercase.

    Args:
        text (str): The raw text containing HTML.

    Returns:
        str: The cleaned, lowercased text.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags using regex
    # Replace with space to avoid merging words (e.g. "end</p><p>start" -> "end start")
    text = re.sub(r"<.*?>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace resulting from HTML stripping
    text = re.sub(r"\s+", " ", text).strip()

    return text


class Timer(contextlib.ContextDecorator):
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Starting...")
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] Completed in {elapsed_time:.6f} seconds.")
        sys.stdout.flush()
        return False  # Propagate exceptions if any
