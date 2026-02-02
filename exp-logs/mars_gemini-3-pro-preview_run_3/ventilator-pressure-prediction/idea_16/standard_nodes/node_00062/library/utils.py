import os
import random
import glob
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_cache(working_dir=Config.WORKING_DIR):
    """
    Removes all .npy files from the specified working directory to invalidate stale cache.
    This ensures that the pipeline does not accidentally use data from previous
    incompatible runs or experiments.

    Args:
        working_dir (str): The directory to clean. Defaults to Config.WORKING_DIR.
    """
    if not os.path.exists(working_dir):
        return

    # Find all .npy files in the directory
    pattern = os.path.join(working_dir, "*.npy")
    files = glob.glob(pattern)

    for file_path in files:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error removing {file_path}: {e}")
