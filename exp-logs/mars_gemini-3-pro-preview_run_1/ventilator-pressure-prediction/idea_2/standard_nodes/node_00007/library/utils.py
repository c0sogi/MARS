import os
import random
import hashlib
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_hash() -> str:
    """
    Generates a unique MD5 hash based on the current feature configuration in Config.
    This hash is used to name cached files, ensuring that any change in feature
    definitions (e.g., adding a lag feature) or data settings (e.g., debug mode)
    invalidates old caches.

    Returns:
        str: The MD5 hash of the configuration state.
    """
    # Collect configuration parameters that affect data structure/content
    config_state = {
        "CONT_FEATURES": Config.CONT_FEATURES,
        "CAT_FEATURES": Config.CAT_FEATURES,
        "R_VALUES": Config.R_VALUES,
        "C_VALUES": Config.C_VALUES,
        "SEQ_LEN": Config.SEQ_LEN,
        "MASK_EXPIRATORY_PHASE": Config.MASK_EXPIRATORY_PHASE,
        "INPUT_DIM": Config.INPUT_DIM,
        # Include DEBUG flags to prevent loading a small debug cache during a full run
        "DEBUG": Config.DEBUG,
        "DEBUG_SAMPLE_SIZE": Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None,
    }

    # Convert to string representation, sorting keys for determinism
    config_str = str(sorted(config_state.items()))

    # Generate MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
