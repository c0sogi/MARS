import os
import random
import numpy as np
import torch
import re
import unicodedata
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_semiotic(text: str) -> bool:
    """
    Determines if a token is 'semiotic' (contains digits or Latin characters).
    This is used to decide whether to route a token to the Tier 2 Transformer
    or rely on the Tier 1 HFBB / Identity fallback.

    Args:
        text (str): The input token text.

    Returns:
        bool: True if the text contains any digit or ASCII letter, False otherwise.
    """
    if not isinstance(text, str):
        return False
    # Regex to check for any digit (\d) or Latin letter ([a-zA-Z])
    # This covers numbers ($3.16), dates (2012), and transliterated entities (Julius).
    return bool(re.search(r"[\d]|[a-zA-Z]", text))


def clean_text(text: str) -> str:
    """
    Performs standard text normalization on a string.
    Uses NFKC unicode normalization to ensure consistency (e.g., combining characters,
    width variations).

    Args:
        text (str): The raw input string.

    Returns:
        str: The normalized string.
    """
    if not isinstance(text, str):
        return str(text)

    # Normalize unicode characters to NFKC form
    text = unicodedata.normalize("NFKC", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text
