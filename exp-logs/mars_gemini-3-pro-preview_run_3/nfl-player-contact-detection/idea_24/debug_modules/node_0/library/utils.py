import hashlib
import json
import logging
import os
import random
import sys
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


def setup_logger(name="idea_24", level=logging.INFO):
    """
    Configures and returns a logger instance that outputs to stdout.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_config_hash():
    """
    Generates a unique MD5 hash based on the critical feature engineering and
    model configuration parameters in library.config.Config.

    This hash is used to name cache files. If the configuration (features, lags,
    hyperparameters) changes, the hash changes, forcing a re-computation of
    features or models.

    Returns:
        str: A hexadecimal MD5 hash string representing the current configuration.
    """
    # Extract configuration elements that affect data processing and training logic
    config_dict = {
        "STREAM_A_LAGS": Config.STREAM_A_LAGS,
        "STREAM_A_VISUAL_LAGS": Config.STREAM_A_VISUAL_LAGS,
        "STREAM_A_LAG_COLS": Config.STREAM_A_LAG_COLS,
        "STREAM_A_FEATURES": Config.STREAM_A_FEATURES,
        "STREAM_B_FEATURES": Config.STREAM_B_FEATURES,
        "NEG_POS_RATIO": Config.NEG_POS_RATIO,
        "XGB_STREAM_A": Config.XGB_STREAM_A,
        "XGB_STREAM_B": Config.XGB_STREAM_B,
        "SEED": Config.SEED,
    }

    # Serialize to JSON with sorted keys to ensure determinism
    # default=str handles any types that might not be natively JSON serializable
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Generate MD5 hash
    md5_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    return md5_hash
