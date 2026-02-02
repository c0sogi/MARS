import os
import random
import numpy as np
import torch
import hashlib
import json


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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

    # Enforce deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_hash(config) -> str:
    """
    Generates a unique MD5 hash based on the configuration parameters that affect
    data processing, feature engineering, and dataset structure.

    This hash is used to version cached datasets. If any relevant parameter
    (like the list of features or sequence length) changes, the hash changes,
    prompting a rebuild of the dataset cache.

    Args:
        config: The Config class or object containing configuration attributes.

    Returns:
        str: A hexadecimal MD5 hash string.
    """
    # Extract parameters that impact data generation
    # We deliberately exclude model training params (LR, EPOCHS) as they don't change the data
    config_dict = {
        "FEATURE_COLS": config.FEATURE_COLS,
        "CONTEXT_FEATURES": config.CONTEXT_FEATURES,
        "SEQ_LEN": config.SEQ_LEN,
        "DEBUG": config.DEBUG,
        "DEBUG_SAMPLE_SIZE": config.DEBUG_SAMPLE_SIZE,
        "RAW_COLS": config.RAW_COLS,
        "TRAIN_PATH": config.TRAIN_PATH,
        "VAL_PATH": config.VAL_PATH,
        "TEST_PATH": config.TEST_PATH,
        "TARGET_COL": config.TARGET_COL,
        "ID_COL": config.ID_COL,
        "BREATH_ID_COL": config.BREATH_ID_COL,
    }

    # Serialize to JSON string with sorted keys to ensure deterministic ordering
    config_str = json.dumps(config_dict, sort_keys=True)

    # Generate MD5 hash
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    return config_hash
