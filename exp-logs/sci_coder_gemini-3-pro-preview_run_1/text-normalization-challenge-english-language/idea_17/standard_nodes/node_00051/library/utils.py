import os
import random
import numpy as np
import torch
import re
from typing import List
from library.config import REGEX_PATTERNS, SEED

# Pre-compile regex patterns for efficiency
# This ensures we don't re-compile the regex engine for every single token in the dataset.
COMPILED_REGEX_PATTERNS = [
    (re.compile(pattern), name) for pattern, name in REGEX_PATTERNS
]


def set_seed(seed: int = SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can be slower, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_regex_features(text: str) -> List[int]:
    """
    Extracts binary features based on the pre-defined regex patterns in config.

    Args:
        text (str): The input token text.

    Returns:
        List[int]: A list of binary values (0 or 1) corresponding to each regex pattern.
                   Order matches REGEX_PATTERNS in config.
    """
    # Handle potential non-string inputs gracefully, though dataset should be clean
    if not isinstance(text, str):
        text = str(text)

    features = []
    for pattern, _ in COMPILED_REGEX_PATTERNS:
        # re.search checks for a match anywhere in the string, but patterns in config
        # often use anchors (^, $) to enforce full string matching where necessary.
        if pattern.search(text):
            features.append(1)
        else:
            features.append(0)

    return features
