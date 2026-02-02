import os
import random
import re
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def is_semiotic(text: str) -> bool:
    """
    Determines if a token is 'semiotic' (requires complex normalization)
    by checking for the presence of digits or Latin characters.

    This function acts as the gate for the Tier 2 Transformer model.

    Args:
        text (str): The input token text.

    Returns:
        bool: True if the text matches the GATE_REGEX (contains digits or Latin chars),
              False otherwise.
    """
    if not isinstance(text, str):
        return False

    # Use the regex defined in Config to determine if Tier 2 (Transformer) is needed
    return bool(re.search(Config.GATE_REGEX, text))
